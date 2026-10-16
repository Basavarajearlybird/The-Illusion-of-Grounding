"""LLaVA-1.6 / LLaVA-NeXT adapters.

7B and 13B are both registered so the paper can report a scale contrast within a
single architecture -- which is what lets the Pareto-frontier claim ("bigger
models are more easily tricked by cross-modal conflict") be tested with model
family held constant rather than confounded across vendors.
"""

from __future__ import annotations

from gib.adapters.hf_vlm import HFVLMAdapter
from gib.adapters.registry import register_adapter


class LLaVANextAdapter(HFVLMAdapter):
    """LLaVA-NeXT uses a Vicuna/Mistral chat template with a single <image> tag."""

    name = "llava-1.6-7b"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    version = "1.0"

    def format_prompt(self, text: str) -> str:
        return f"[INST] <image>\n{text} [/INST]"


@register_adapter("llava-1.6-7b")
def _llava_7b(**kw):
    return LLaVANextAdapter(**kw)


@register_adapter("llava-1.6-13b")
def _llava_13b(**kw):
    a = LLaVANextAdapter(model_id="llava-hf/llava-v1.6-vicuna-13b-hf", **kw)
    a.name = "llava-1.6-13b"
    # The Vicuna-tuned checkpoints use a different chat template from the
    # Mistral-tuned 7B; getting this wrong degrades the model in a way that
    # would be misread as fragility.
    a.format_prompt = lambda text: (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers.\n"
        f"USER: <image>\n{text} ASSISTANT:"
    )
    return a
