"""GPT-4o (and successors) via the OpenAI API.

Track B. The API does expose top-k logprobs, but only over the top ~20 tokens of
each position, which is not enough to score an arbitrary candidate answer under
teacher forcing. Estimating the Trust Index from a truncated distribution would
produce a number that silently means something different from Track A's, so this
adapter reports logprobs=False and the runner routes it to sampling-based
estimation instead.
"""

from __future__ import annotations

from typing import Any

from gib.adapters.api_base import APIAdapter, encode_image_b64
from gib.adapters.registry import register_adapter


class OpenAIVLMAdapter(APIAdapter):
    name = "gpt-4o"
    version = "1.0"
    env_key = "OPENAI_API_KEY"
    model_id = "gpt-4o"

    def __init__(self, model_id: str | None = None, **kw):
        super().__init__(**kw)
        self.model_id = model_id or self.model_id

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.require_key())
        return self._client

    def _call(self, image: str, prompt: str, temperature: float) -> tuple[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image:
            b64, mime = encode_image_b64(image)
            content.append(
                {
                    "type": "image_url",
                    # `detail: high` so the model is not handed a downsampled
                    # image -- that would add an uncontrolled degradation on top
                    # of the one Axis 2 applied deliberately.
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                }
            )
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or "", {
            "usage": getattr(resp, "usage", None) and resp.usage.model_dump(),
            "model": resp.model,
        }


@register_adapter("gpt-4o")
def _gpt4o(**kw):
    return OpenAIVLMAdapter(**kw)


@register_adapter("gpt-4o-mini")
def _gpt4o_mini(**kw):
    a = OpenAIVLMAdapter(model_id="gpt-4o-mini", **kw)
    a.name = "gpt-4o-mini"
    return a
