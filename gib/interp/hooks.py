"""Layer 3 -- attention capture and Visual Attention Mass.

In LLaVA-style architectures image patches enter the language model as projected
*visual tokens*, so "cross-modal attention" is the decoder's self-attention over
the visual-token span. Qwen-VL differs in detail but exposes the same quantity.
That equivalence is what lets one definition serve the whole open zoo:

    VAM = (sum of attention to visual-token positions)
          / (sum of attention to all positions)

computed per layer and per head, averaged over the answer-generation steps.

Two practical constraints shape this module. Hooks slow inference badly and
produce very large tensors, so (a) this layer runs as a **separate branch** on a
curated probe set rather than inside the main eval loop, and (b) the reduction to
VAM happens *inside* the hook -- storing raw attention for 32 layers x 32 heads x
600 positions per item would be gigabytes per model for no added information.

Attention capture also requires eager attention: fused/SDPA/Flash kernels never
materialise the attention matrix, and silently return ``None`` instead. The
adapter defaults to ``attn_implementation="eager"`` for this reason, and
:func:`assert_eager_attention` fails loudly rather than letting a run produce
empty captures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class AttentionCaptureError(RuntimeError):
    pass


def assert_eager_attention(adapter) -> None:
    impl = getattr(adapter, "attn_implementation", None)
    if impl not in (None, "eager"):
        raise AttentionCaptureError(
            f"{adapter.name} is loaded with attn_implementation={impl!r}; attention "
            f"weights are not materialised. Reload with attn_implementation='eager'."
        )


@dataclass
class VAMCapture:
    """Per-item Visual Attention Mass, reduced from raw attention."""

    variant_id: str
    model: str
    vam_per_layer: np.ndarray        # (n_layers,)  averaged over heads and steps
    vam_per_layer_head: np.ndarray   # (n_layers, n_heads)
    vam_overall: float
    visual_span: tuple[int, int]
    n_steps: int
    n_visual_tokens: int
    meta: dict = field(default_factory=dict)

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vam_per_layer=self.vam_per_layer,
            vam_per_layer_head=self.vam_per_layer_head,
            vam_overall=np.array([self.vam_overall]),
            visual_span=np.array(self.visual_span),
            n_steps=np.array([self.n_steps]),
            n_visual_tokens=np.array([self.n_visual_tokens]),
        )

    @staticmethod
    def from_npz(path: Path, variant_id: str = "", model: str = "") -> VAMCapture:
        z = np.load(path)
        return VAMCapture(
            variant_id=variant_id or path.stem,
            model=model,
            vam_per_layer=z["vam_per_layer"],
            vam_per_layer_head=z["vam_per_layer_head"],
            vam_overall=float(z["vam_overall"][0]),
            visual_span=tuple(int(v) for v in z["visual_span"]),
            n_steps=int(z["n_steps"][0]),
            n_visual_tokens=int(z["n_visual_tokens"][0]),
        )


def vam_from_generation(attn_capture: dict, variant_id: str, model: str, meta: dict | None = None):
    """Build a :class:`VAMCapture` from the adapter's packed attention.

    The adapter already reduces to ``(steps, layers, heads)`` visual-mass ratios
    during generation; this function only aggregates over steps.
    """
    if not attn_capture or "vam_per_step_layer_head" not in attn_capture:
        return None

    arr = np.asarray(attn_capture["vam_per_step_layer_head"], dtype=np.float32)
    if arr.ndim != 3 or arr.size == 0:
        return None

    span = tuple(int(v) for v in attn_capture.get("visual_span", (0, 0)))
    per_layer_head = arr.mean(axis=0)         # (L, H)
    per_layer = per_layer_head.mean(axis=1)   # (L,)

    return VAMCapture(
        variant_id=variant_id,
        model=model,
        vam_per_layer=per_layer,
        vam_per_layer_head=per_layer_head,
        vam_overall=float(per_layer_head.mean()),
        visual_span=span,
        n_steps=int(arr.shape[0]),
        n_visual_tokens=max(0, span[1] - span[0]),
        meta=meta or {},
    )


# --------------------------------------------------------------------------- #
# Spatial attention maps (for the qualitative heatmap panels)
# --------------------------------------------------------------------------- #


def visual_attention_grid(
    attn_capture: dict, grid_hw: tuple[int, int] | None = None
) -> np.ndarray | None:
    """Reshape per-visual-token attention into a 2D grid for overlaying on the image.

    Returns ``None`` when the visual-token count is not a perfect square and no
    grid shape is supplied. LLaVA-NeXT's dynamic tiling can emit a non-square
    token count, and guessing a shape would produce a heatmap that looks
    convincing while being spatially wrong -- worse than no figure.
    """
    per_token = attn_capture.get("per_visual_token")
    if per_token is None:
        return None
    v = np.asarray(per_token, dtype=np.float32).ravel()
    n = v.size
    if grid_hw is None:
        side = int(round(np.sqrt(n)))
        if side * side != n:
            return None
        grid_hw = (side, side)
    if grid_hw[0] * grid_hw[1] != n:
        return None
    return v.reshape(grid_hw)


class AttentionHookManager:
    """Register/remove forward hooks on the LM's self-attention modules.

    The adapter's ``output_attentions=True`` path covers the common case; this
    manager exists for architectures that do not propagate that flag through
    ``generate``, and for capturing per-visual-token spatial maps that the packed
    reduction discards.
    """

    def __init__(self, model, layer_pattern: str = "self_attn"):
        self.model = model
        self.layer_pattern = layer_pattern
        self._handles: list = []
        self.buffer: list = []

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def register(self) -> int:
        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                # HF attention modules return (hidden, attn_weights, ...) when
                # output_attentions is on; anything else means weights are not
                # available and must not be silently treated as zeros.
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self.buffer.append((idx, output[1].detach()))

            return hook

        for i, (name, module) in enumerate(self.model.named_modules()):
            if name.endswith(self.layer_pattern):
                self._handles.append(module.register_forward_hook(make_hook(i)))
        if not self._handles:
            raise AttentionCaptureError(
                f"no modules matched {self.layer_pattern!r}; inspect model.named_modules()"
            )
        return len(self._handles)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.buffer.clear()
