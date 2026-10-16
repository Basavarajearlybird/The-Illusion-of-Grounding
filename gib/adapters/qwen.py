"""Qwen-VL adapters.

The proposal names Qwen-VL, and Qwen2.5-VL is registered alongside it. The
architecture's reasoning is worth keeping in the code: by 2026 a model zoo whose
newest open model is two generations old invites a reviewer objection, and the
marginal cost of covering the current generation is one adapter file. The
original checkpoint stays registered for continuity with the proposal.
"""

from __future__ import annotations

from gib.adapters.hf_vlm import HFVLMAdapter
from gib.adapters.registry import register_adapter


class QwenVLAdapter(HFVLMAdapter):
    """Qwen2/2.5-VL ChatML template.

    Qwen wraps image placeholders in ``<|vision_start|>...<|vision_end|>`` and
    expands ``<|image_pad|>`` into the visual-token span; the generic span
    locator in :class:`HFVLMAdapter` finds it via ``config.image_token_id``.
    """

    name = "qwen2.5-vl-7b"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    version = "1.0"

    def format_prompt(self, text: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _prepare(self, image_path: str | None, text: str):
        """Qwen's processor needs the chat template applied before tokenisation
        so the vision placeholders expand to the right patch count."""
        if image_path is None:
            inputs = self.processor(text=[text], return_tensors="pt")
            return {k: v.to(self.model.device) for k, v in inputs.items()}
        img = self._load_image(image_path)
        inputs = self.processor(
            text=[self.format_prompt(text)], images=[img], return_tensors="pt"
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}


@register_adapter("qwen2.5-vl-7b")
def _qwen25(**kw):
    return QwenVLAdapter(**kw)


@register_adapter("qwen-vl-chat")
def _qwen_v1(**kw):
    """The original Qwen-VL-Chat named in the proposal."""
    a = QwenVLAdapter(model_id="Qwen/Qwen-VL-Chat", **kw)
    a.name = "qwen-vl-chat"
    a.format_prompt = lambda text: f"Picture 1: <img>image</img>\n{text}"
    return a
