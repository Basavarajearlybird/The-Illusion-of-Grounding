"""Simulated adapters -- pipeline verification ONLY.

=============================================================================
 THESE ARE NOT MODELS. THEY PRODUCE NO SCIENTIFIC RESULT OF ANY KIND.
 Every row they generate is stamped ``simulated=True``, every parquet they
 write lands under a ``sim-`` model name, and every figure rendered from
 their output carries a "SIMULATED DATA" watermark. Do not put a number
 produced by this module into the paper.
=============================================================================

They exist for one reason. The architecture's build order says to write each
figure function the same week its data first exists, and to de-risk the judge
cascade with a tracer-bullet run before scaling. Both require *something* to
push through the pipe before an A100 or an API key is available. A simulator
whose behaviour responds to the stressors -- accuracy decaying with distractor
count, collapsing past a blur threshold, and resolving conflicts with a
model-specific text bias -- exercises every downstream code path (scoring, the
taxonomy classifier, ECE, decay fitting, the heatmap, the Pareto frontier)
against data with the right *shape*.

Behaviour is a deterministic function of ``(model, variant_id)``, so a simulated
run is exactly reproducible and cache behaviour can itself be tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gib.adapters.base import BaseAdapter, GenerationResult
from gib.adapters.registry import register_adapter
from gib.seeds import np_rng_for, rng_for


@dataclass
class SimProfile:
    """A synthetic model personality.

    The profiles are staggered so the downstream analyses have something to find:
    ``sim-frontier-a`` is the strong-but-fragile case (high clean accuracy, high
    text bias) and ``sim-small-open`` the weak-but-stubborn one, which is exactly
    the configuration that makes the Pareto frontier slope downward. That is a
    property of these fixtures, not a prediction about real models.
    """

    clean_acc: float
    decay_lambda: float       # accuracy decay per distractor doc
    noise_floor: float        # asymptotic accuracy under heavy noise
    blur_sigma_50: float      # sigma at which accuracy halves
    crop_tolerance: float     # visible fraction at which accuracy halves
    text_bias: float          # P(follow text) under cross-modal conflict
    overconfidence: float     # gap between stated confidence and accuracy
    collapse_rate: float      # P(incoherent output | failure)
    track: str = "A"


PROFILES: dict[str, SimProfile] = {
    "sim-frontier-a": SimProfile(0.91, 0.115, 0.30, 6.0, 0.14, 0.74, 0.22, 0.06, "B"),
    "sim-frontier-b": SimProfile(0.88, 0.098, 0.34, 7.0, 0.12, 0.68, 0.18, 0.07, "B"),
    "sim-open-13b": SimProfile(0.79, 0.135, 0.24, 4.5, 0.19, 0.61, 0.26, 0.11, "A"),
    "sim-open-7b": SimProfile(0.72, 0.152, 0.20, 3.5, 0.24, 0.55, 0.29, 0.15, "A"),
    "sim-small-open": SimProfile(0.63, 0.104, 0.22, 2.8, 0.30, 0.41, 0.31, 0.19, "A"),
}

GIBBERISH = [
    "the image contains several items of interest",
    "it is difficult to determine from the given information",
    "based on the passage, the answer relates to the context",
    "there appear to be multiple objects present",
]


class SimulatedAdapter(BaseAdapter):
    """A stressor-responsive fake VLM."""

    version = "sim-1.0"
    is_simulated = True

    def __init__(self, name: str, profile: SimProfile, **_: Any):
        self.name = name
        self.profile = profile
        self.capabilities = {
            "logprobs": profile.track == "A",
            "attention": profile.track == "A",
            "batch": True,
            "captioning": True,
        }

    # -- the behavioural model ------------------------------------------------ #

    def _p_correct(self, meta: dict) -> float:
        """P(correct) given the stressor and dose encoded in the variant."""
        p = self.profile.clean_acc
        axis = meta.get("axis", "clean")
        floor = self.profile.noise_floor

        if axis in ("noise", "combined"):
            d = float(meta.get("n_distractors", 0) or 0)
            p = floor + (p - floor) * math.exp(-self.profile.decay_lambda * d)

        if axis in ("visual", "combined"):
            deg = meta.get("degradation", "none")
            param = float(meta.get("degradation_param", 0) or 0)
            if deg == "gaussian_blur" and param > 0:
                # Logistic falloff around the profile's sigma_50.
                p *= 1.0 / (1.0 + (param / self.profile.blur_sigma_50) ** 2.2)
            elif deg == "motion_blur" and param > 0:
                p *= 1.0 / (1.0 + (param / (self.profile.blur_sigma_50 * 4)) ** 1.8)
            elif deg == "crop":
                p *= 1.0 / (1.0 + (self.profile.crop_tolerance / max(param, 1e-3)) ** 1.6)

        if axis == "conflict":
            # Under a pure clash, "correct" means resisting the text.
            p = 1.0 - self.profile.text_bias

        return max(0.02, min(0.99, p))

    def _respond(self, meta: dict) -> tuple[str, float]:
        """Returns (answer, confidence)."""
        rng = rng_for("sim", self.name, meta.get("variant_id", ""))
        p = self._p_correct(meta)
        visual = meta.get("answer_visual", "")
        textual = meta.get("answer_textual", "")

        if rng.random() < p:
            answer = visual
            conf = min(0.99, p + self.profile.overconfidence * (1 - p))
        else:
            axis = meta.get("axis", "clean")
            if rng.random() < self.profile.collapse_rate:
                answer = rng.choice(GIBBERISH)              # Reasoning Collapse
            elif textual and axis in ("conflict", "noise", "combined"):
                answer = textual                             # Knowledge Inertia
            elif axis in ("visual", "combined"):
                answer = meta.get("distractor_answer") or "unknown"  # Sensory Capture
            else:
                answer = meta.get("distractor_answer") or "unknown"
            # The defining pathology the Robustness Score penalises: confidently
            # wrong. Confidence stays high even as accuracy falls.
            conf = min(0.99, p + self.profile.overconfidence)

        return answer, conf

    # -- adapter interface ----------------------------------------------------- #

    def generate(
        self,
        image: str,
        question: str,
        docs: list[str],
        *,
        temperature: float = 0.0,
        capture_attention: bool = False,
        meta: dict | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        meta = dict(meta or {})
        if temperature > 0:
            # Sampling jitter, so Track B self-consistency is not degenerate.
            meta["variant_id"] = f"{meta.get('variant_id','')}#{kwargs.get('sample_idx', 0)}"
        answer, conf = self._respond(meta)
        n_tok = max(1, len(answer.split()))
        return GenerationResult(
            answer=answer,
            latency=0.001,
            token_logprobs=[math.log(max(conf, 1e-6))] * n_tok if self.capabilities["logprobs"] else None,
            attn_capture=self._fake_attention(meta, answer) if capture_attention else None,
            raw={"simulated": True, "p_correct": self._p_correct(meta), "confidence": conf},
        )

    def _fake_attention(
        self, meta: dict, answer: str, n_layers: int = 32, n_heads: int = 32
    ) -> dict[str, Any]:
        """Synthetic VAM tensors shaped like a real capture.

        Encodes the hypothesis Layer 3 exists to test: visual attention mass is
        substantially lower on items the model resolved by following the text
        than on clean items it got right, and the drop is concentrated in the
        middle-to-late layers where cross-modal integration happens.

        This is a fixture. It is what makes the paired-test code, the FDR
        correction, and the head-level map runnable before any GPU exists -- it
        is not evidence for the hypothesis it encodes.
        """
        import numpy as np

        rng = np_rng_for("sim_attn", self.name, meta.get("variant_id", ""))
        axis = meta.get("axis", "clean")
        followed_text = bool(meta.get("answer_textual")) and answer == meta.get("answer_textual")

        # Baseline VAM rises then falls across depth, as in real decoders.
        layers = np.arange(n_layers)
        base = 0.34 * np.exp(-((layers - n_layers * 0.35) ** 2) / (2 * (n_layers * 0.3) ** 2)) + 0.06

        if followed_text:
            collapse = 0.62          # the Knowledge Inertia case
        elif axis == "conflict":
            collapse = 0.22          # resisted, but the clash still costs some mass
        else:
            collapse = 0.0
        # Concentrated in the middle-to-late layers rather than uniform.
        depth_weight = 1.0 / (1.0 + np.exp(-(layers - n_layers * 0.45) / (n_layers * 0.10)))
        per_layer = base * (1.0 - collapse * depth_weight)

        n_steps = max(1, min(8, len(answer.split())))
        arr = (
            per_layer[None, :, None]
            * rng.normal(1.0, 0.12, size=(n_steps, n_layers, n_heads))
        )
        # A minority of heads carry most of the visual mass, as observed in real
        # models -- this is what makes the head-level scatter non-degenerate.
        head_gain = rng.lognormal(0.0, 0.45, size=n_heads)
        arr = arr * head_gain[None, None, :]

        return {
            "vam_per_step_layer_head": np.clip(arr, 0.0, 1.0).astype(np.float32),
            "visual_span": np.array([35, 35 + 576], dtype=np.int32),
        }

    def score_answer(
        self, image: str, question: str, docs: list[str], candidate: str, *, meta: dict | None = None, **kw
    ) -> float:
        """Log-prob that makes the Trust Index come out at the profile's bias."""
        meta = dict(meta or {})
        visual, textual = meta.get("answer_visual", ""), meta.get("answer_textual", "")
        bias = self.profile.text_bias if meta.get("axis") == "conflict" else 0.25
        jitter = rng_for("sim_score", self.name, meta.get("variant_id", ""), candidate).uniform(-0.06, 0.06)
        if candidate == textual and textual:
            p = bias + jitter
        elif candidate == visual:
            p = (1 - bias) + jitter
        else:
            p = 0.02
        return math.log(max(1e-6, min(0.999, p)))

    def token_logprobs_with_without_context(
        self, image: str, question: str, docs: list[str], target: str, *, meta: dict | None = None, **kw
    ) -> tuple[list[float], list[float]]:
        """CSR primitive: the context lifts the injected falsehood's probability."""
        meta = dict(meta or {})
        textual = meta.get("answer_textual", "")
        rng = rng_for("sim_csr", self.name, meta.get("variant_id", ""), target)
        n = max(1, len(target.split()))
        base = [math.log(rng.uniform(0.05, 0.20)) for _ in range(n)]
        if target and target == textual:
            lift = self.profile.text_bias * 2.0
            with_ctx = [b + lift for b in base]
        else:
            with_ctx = [b + rng.uniform(-0.1, 0.2) for b in base]
        return with_ctx, base

    def caption(self, image: str, *, meta: dict | None = None, **kw) -> str:
        """Captions that hallucinate objects at a blur-dependent rate, so CHAIR
        has a real signal to measure."""
        meta = dict(meta or {})
        rng = rng_for("sim_caption", self.name, meta.get("variant_id", ""))
        present = list(meta.get("gt_objects") or [])
        pool = list(meta.get("candidate_objects") or [])
        p = self._p_correct(meta)
        mentioned = [o for o in present if rng.random() < p]
        n_halluc = sum(1 for _ in range(3) if rng.random() < (1 - p) * 0.8)
        halluc = [o for o in rng.sample(pool, min(n_halluc, len(pool))) if o not in present]
        items = mentioned + halluc
        rng.shuffle(items)
        if not items:
            return "The image is unclear."
        return "The image shows " + ", ".join(f"a {i}" for i in items) + "."


def _make(name: str):
    def factory(**kw):
        return SimulatedAdapter(name, PROFILES[name], **kw)

    return factory


for _name in PROFILES:
    register_adapter(_name)(_make(_name))
