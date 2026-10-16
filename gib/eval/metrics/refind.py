"""REFIND / CSR -- token-level Context Sensitivity Ratio.

Lee & Yu, REFIND (SemEval-2025 Task 3, "Mu-SHROOM"): detect hallucination in
retrieval-augmented generation by measuring how much each output token's
probability shifts when the retrieved context is present versus absent.

    CSR(t) = P(t | context, image, question) / P(t | image, question)

**Why this metric matters most here.** CSR is the strongest bridge between the
project's metrics and its taxonomy. High CSR on tokens that match the *injected
falsehood* is direct, quantitative evidence of Knowledge Inertia: it shows the
model's tokens are being driven by the poisoned context rather than merely
coinciding with it. Every other signal in the benchmark is behavioural -- CSR is
mechanistic, and it is measured at the output rather than requiring the attention
hooks of Layer 3.

CSR needs token probabilities, so it runs natively on Track A. On Track B it runs
through a **proxy scorer** -- an open model re-scores the closed model's emitted
string. That is a standard and citable workaround, but it measures the proxy's
context sensitivity toward the closed model's output, not the closed model's own.
:func:`csr_for_target` therefore stamps ``proxy=True`` on those rows, and Layer 5
labels them in every figure they appear in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CSRResult:
    csr_mean: float                 # mean log-ratio across target tokens
    csr_max: float
    per_token: list[float] = field(default_factory=list)
    n_tokens: int = 0
    proxy: bool = False
    proxy_model: str = ""

    def as_dict(self) -> dict:
        return {
            "csr_mean": round(self.csr_mean, 4),
            "csr_max": round(self.csr_max, 4),
            "csr_n_tokens": self.n_tokens,
            "csr_proxy": self.proxy,
            "csr_proxy_model": self.proxy_model,
        }


def csr_from_logprobs(
    with_context: list[float], without_context: list[float]
) -> tuple[float, float, list[float]]:
    """Per-token log CSR from paired logprob sequences.

    The ratio is kept in log space: probabilities of individual tokens routinely
    underflow, and log-space differences are what the downstream statistics
    (means, Wilcoxon tests) should operate on anyway. A value of 0 means the
    context did not move the token; positive means the context raised it.
    """
    n = min(len(with_context), len(without_context))
    if n == 0:
        return float("nan"), float("nan"), []
    per_token = [float(with_context[i] - without_context[i]) for i in range(n)]
    return float(np.mean(per_token)), float(np.max(per_token)), per_token


def csr_for_target(
    adapter,
    *,
    image: str,
    question: str,
    docs: list[str],
    target: str,
    meta: dict | None = None,
    proxy_adapter=None,
) -> CSRResult:
    """CSR of ``target`` under ``docs``.

    ``target`` is normally ``answer_textual`` -- the injected falsehood -- so a
    high value is precisely the Knowledge Inertia evidence described above.
    """
    scorer = adapter
    proxy = False
    proxy_model = ""
    if not adapter.capabilities.get("logprobs"):
        if proxy_adapter is None:
            return CSRResult(float("nan"), float("nan"), [], 0, False, "")
        scorer = proxy_adapter
        proxy = True
        proxy_model = getattr(proxy_adapter, "name", "proxy")

    kw = {"meta": meta} if meta is not None else {}
    with_ctx, without_ctx = scorer.token_logprobs_with_without_context(
        image, question, docs, target, **kw
    )
    mean, mx, per_token = csr_from_logprobs(with_ctx, without_ctx)
    return CSRResult(mean, mx, per_token, len(per_token), proxy, proxy_model)


def inertia_evidence(
    csr_textual: CSRResult, csr_visual: CSRResult
) -> float:
    """Contrast: how much more the context lifts the falsehood than the truth.

    Reported alongside the raw CSR because a model whose probabilities all rise
    when *any* context is added is not exhibiting inertia -- it is exhibiting a
    generic context effect. The differential is the part that is diagnostic.
    """
    if math.isnan(csr_textual.csr_mean) or math.isnan(csr_visual.csr_mean):
        return float("nan")
    return float(csr_textual.csr_mean - csr_visual.csr_mean)


def aggregate_csr(rows: list[dict]) -> dict:
    """Corpus-level CSR summary, split by taxonomy label.

    The headline number for the paper is the contrast between the Knowledge
    Inertia rows and the correct rows: if the taxonomy classifier and CSR are
    both measuring what they claim to, inertia-labelled items must show
    substantially higher CSR on the injected falsehood.
    """
    from gib.schema import FailureMode

    def _mean(vals):
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    inertia = [r for r in rows if r.get("failure_mode") == FailureMode.KNOWLEDGE_INERTIA.value]
    correct = [r for r in rows if r.get("failure_mode") == FailureMode.CORRECT.value]
    return {
        "csr_inertia": _mean([r.get("csr") for r in inertia]),
        "csr_correct": _mean([r.get("csr") for r in correct]),
        "n_inertia": len(inertia),
        "n_correct": len(correct),
        "csr_gap": _mean([r.get("csr") for r in inertia]) - _mean([r.get("csr") for r in correct]),
    }
