"""Adapter registry.

Adapters register themselves at import time; the registry imports the adapter
modules lazily and tolerates individual import failures. That tolerance is
deliberate: a machine with no ``torch`` should still be able to run the API
models, and a machine with no API SDKs should still be able to run the local
ones. A hard import at module scope would make the whole zoo unusable whenever
any one backend's dependency was missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gib.adapters.base import BaseAdapter

_ADAPTERS: dict[str, Callable[..., BaseAdapter]] = {}
_IMPORT_ERRORS: dict[str, str] = {}
_LOADED = False


def register_adapter(name: str):
    def deco(factory):
        _ADAPTERS[name] = factory
        return factory

    return deco


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod in (
        "gib.adapters.simulated",
        "gib.adapters.llava",
        "gib.adapters.qwen",
        "gib.adapters.openai_vlm",
        "gib.adapters.gemini_vlm",
    ):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            _IMPORT_ERRORS[mod] = f"{type(exc).__name__}: {exc}"


def get_adapter(name: str, **kwargs: Any) -> BaseAdapter:
    _load_all()
    if name not in _ADAPTERS:
        hint = ""
        if _IMPORT_ERRORS:
            hint = "\n  (some adapter modules failed to import: " + \
                   "; ".join(f"{k}: {v}" for k, v in _IMPORT_ERRORS.items()) + ")"
        raise KeyError(f"unknown model {name!r}. Available: {sorted(_ADAPTERS)}{hint}")
    return _ADAPTERS[name](**kwargs)


def available_models() -> list[str]:
    _load_all()
    return sorted(_ADAPTERS)


def import_errors() -> dict[str, str]:
    _load_all()
    return dict(_IMPORT_ERRORS)


def is_simulated(name: str) -> bool:
    return name.startswith("sim-")


# --------------------------------------------------------------------------- #
# Text-only LLM helper (judge cascade, QA paraphrasing)
# --------------------------------------------------------------------------- #


def get_text_model(model: str = "claude-opus-4-8") -> Callable[[str], str]:
    """Return ``prompt -> completion`` for a text-only LLM.

    Used by the LLM-as-judge stage and the optional QA paraphraser. Kept separate
    from the VLM registry because the judge must be pinnable independently of the
    zoo -- see the judge-circularity guard in :mod:`gib.eval.judge`.
    """
    if model.startswith("claude"):
        def _anthropic(prompt: str) -> str:
            import anthropic

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return _anthropic

    if model.startswith("gpt"):
        def _openai(prompt: str) -> str:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        return _openai

    raise ValueError(f"no text backend for {model!r}")
