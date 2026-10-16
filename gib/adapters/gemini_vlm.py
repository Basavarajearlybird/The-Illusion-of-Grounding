"""Gemini 1.5 Pro (and successors) via the Google Generative AI API.

Track B, and the strictest case for the dual-track design: Gemini exposes
effectively no token probabilities at all, so *every* probabilistic quantity in
the paper -- Trust Index, confidence, calibration -- has to come from sampling
for this model. That is precisely why Track B exists as a first-class measurement
path rather than as a footnote.
"""

from __future__ import annotations

from typing import Any

from gib.adapters.api_base import APIAdapter
from gib.adapters.registry import register_adapter


class GeminiVLMAdapter(APIAdapter):
    name = "gemini-1.5-pro"
    version = "1.0"
    env_key = "GOOGLE_API_KEY"
    model_id = "gemini-1.5-pro"

    def __init__(self, model_id: str | None = None, **kw):
        kw.setdefault("rpm", 30)  # Gemini's default quotas are tighter
        super().__init__(**kw)
        self.model_id = model_id or self.model_id

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.require_key())
        return self._client

    def _call(self, image: str, prompt: str, temperature: float) -> tuple[str, Any]:
        from google.genai import types

        parts: list[Any] = []
        if image:
            with open(image, "rb") as fh:
                parts.append(types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=prompt))

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=self.max_new_tokens,
                # Safety blocks would otherwise return empty text on innocuous
                # items and be silently scored as wrong answers.
                safety_settings=[
                    types.SafetySetting(category=c, threshold="BLOCK_NONE")
                    for c in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            ),
        )
        text = getattr(resp, "text", None) or ""
        meta = getattr(resp, "usage_metadata", None)
        return text, {"usage": meta.model_dump() if meta else None, "model": self.model_id}


@register_adapter("gemini-1.5-pro")
def _gemini15(**kw):
    return GeminiVLMAdapter(**kw)


@register_adapter("gemini-2.0-flash")
def _gemini2(**kw):
    a = GeminiVLMAdapter(model_id="gemini-2.0-flash", **kw)
    a.name = "gemini-2.0-flash"
    return a
