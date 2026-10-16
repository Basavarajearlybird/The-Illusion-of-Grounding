"""Shared plumbing for closed-source API adapters.

Three things every API backend in this project needs, and none of them are
optional at 60K items:

* **exponential backoff with jitter** -- providers rate-limit hard, and a naive
  retry loop synchronises across workers and makes throttling worse;
* **a token-bucket rate limiter** -- staying under the account's RPM rather than
  discovering it via 429s;
* **the persistent cache** -- see :mod:`gib.adapters.cache`; a crashed run must
  resume for free.

These backends are Track B: GPT-4o exposes only limited top-k logprobs and Gemini
effectively none, so the Trust Index and confidence are estimated by sampling
(Layer 2.2). ``score_answer`` is therefore deliberately left unimplemented rather
than faked from top-k -- a partially-observed distribution would give a Trust
Index that is not comparable with Track A's.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from gib.adapters.base import BaseAdapter, GenerationResult


class RateLimiter:
    """Thread-safe token bucket."""

    def __init__(self, rpm: int = 60):
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.rate = self.capacity / 60.0
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.rate
                )
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


def with_backoff(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.5,
    retry_on: tuple = (Exception,),
) -> Any:
    """Retry with exponential backoff and full jitter."""
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            last = exc
            msg = str(exc).lower()
            # Do not burn the retry budget on errors that will never succeed.
            if any(k in msg for k in ("invalid api key", "authentication", "not found", "permission")):
                raise
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2**attempt)
            time.sleep(random.uniform(0, min(delay, 60.0)))
    raise RuntimeError(f"API call failed after {max_attempts} attempts: {last}") from last


def encode_image_b64(path: str) -> tuple[str, str]:
    """(base64 payload, mime type)."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii"), mime


class APIAdapter(BaseAdapter):
    """Base for closed-source backends. Track B by construction."""

    capabilities = {"logprobs": False, "attention": False, "batch": False, "captioning": True}
    env_key: str = ""

    def __init__(self, *, rpm: int = 60, max_new_tokens: int = 32, api_key: str | None = None):
        self.limiter = RateLimiter(rpm)
        self.max_new_tokens = max_new_tokens
        self.api_key = api_key or os.environ.get(self.env_key, "")
        self._client = None

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"{self.name}: no API key. Set {self.env_key} in the environment, "
                f"or run with --models sim-* for an offline pipeline check."
            )
        return self.api_key

    def _call(self, image: str, prompt: str, temperature: float) -> tuple[str, Any]:
        raise NotImplementedError

    def generate(
        self,
        image: str,
        question: str,
        docs: list[str],
        *,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> GenerationResult:
        from gib.eval.prompt import render_rag_prompt

        prompt = render_rag_prompt(question, docs)
        t0 = time.time()
        self.limiter.acquire()
        answer, raw = with_backoff(lambda: self._call(image, prompt, temperature))
        return GenerationResult(answer=answer.strip(), latency=time.time() - t0, raw=raw)

    def caption(self, image: str, **kwargs: Any) -> str:
        from gib.eval.prompt import CAPTION_PROMPT

        self.limiter.acquire()
        answer, _ = with_backoff(lambda: self._call(image, CAPTION_PROMPT, 0.0))
        return answer.strip()
