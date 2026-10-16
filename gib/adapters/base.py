"""The model adapter interface (Layer 2.1).

Everything in the evaluation engine hinges on this one protocol, so that adding a
model to the zoo is a ~100-line file rather than a fork of the runner.

The ``capabilities`` dict is what makes the dual-track measurement design of
Layer 2.2 possible without branching all over the runner: the runner asks an
adapter what it can do, and routes it to Track A (constrained scoring against
token logprobs) or Track B (sampling-based estimation) accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class GenerationResult:
    """One model response, plus whatever instrumentation the backend supports."""

    answer: str
    latency: float = 0.0
    raw: Any = None

    # Track A only
    token_logprobs: list[float] | None = None
    attn_capture: dict[str, np.ndarray] | None = None

    # Track B only -- populated when the runner samples N responses
    samples: list[str] = field(default_factory=list)

    # Filled by the runner for provenance
    from_cache: bool = False

    @property
    def mean_logprob(self) -> float | None:
        if not self.token_logprobs:
            return None
        return float(np.mean(self.token_logprobs))


@runtime_checkable
class ModelAdapter(Protocol):
    """The contract every backend implements."""

    name: str
    version: str
    capabilities: dict[str, bool]

    def generate(
        self, image: str, question: str, docs: list[str], **kwargs: Any
    ) -> GenerationResult: ...


class BaseAdapter:
    """Shared plumbing: capability defaults, prompt formatting, and the
    optional scoring hooks that Track A backends override."""

    name: str = "base"
    version: str = "1.0"
    capabilities: dict[str, bool] = {
        "logprobs": False,
        "attention": False,
        "batch": False,
        "captioning": False,
    }

    def generate(
        self, image: str, question: str, docs: list[str], **kwargs: Any
    ) -> GenerationResult:
        raise NotImplementedError

    # -- Track A hooks ------------------------------------------------------- #

    def score_answer(
        self, image: str, question: str, docs: list[str], candidate: str, **kwargs: Any
    ) -> float:
        """Sequence log-probability of ``candidate`` under teacher forcing.

        This is the primitive behind the Trust Index: scoring the visual answer
        and the textual answer under the *same* context is what makes their
        ratio interpretable as a modality preference rather than as two
        unrelated generations.
        """
        raise NotImplementedError(f"{self.name} does not support constrained scoring")

    def token_logprobs_with_without_context(
        self, image: str, question: str, docs: list[str], target: str, **kwargs: Any
    ) -> tuple[list[float], list[float]]:
        """Per-token logprobs of ``target`` with and without the retrieved docs.

        This is the REFIND / CSR primitive (Layer 2.3): the shift between the two
        is how much each output token is being driven by the retrieved context.
        """
        raise NotImplementedError(f"{self.name} does not support CSR measurement")

    def caption(self, image: str, **kwargs: Any) -> str:
        """Free-form caption, used by the CHAIR metric on the visual axis."""
        raise NotImplementedError(f"{self.name} does not support captioning mode")

    # -- misc ---------------------------------------------------------------- #

    @property
    def track(self) -> str:
        """A = white-box (logprobs available), B = black-box (sampling only)."""
        return "A" if self.capabilities.get("logprobs") else "B"

    def close(self) -> None:
        pass
