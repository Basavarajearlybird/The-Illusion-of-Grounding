"""The Trust Index and the dual-track measurement design (Layer 2.2).

The proposal's Trust Index and Robustness Score both assume access to token
probabilities, and that is exactly where closed models break a naive design:
GPT-4o exposes only limited top-k logprobs and Gemini effectively none. Rather
than hiding this as a limitation, the architecture resolves it with two
measurement tracks reported side by side, and the paper claims the resolution as
a methodological contribution.

**Track A (white-box).** Constrained scoring. Under teacher forcing, compute the
sequence log-probability of the visual answer and of the textual answer in the
same context, then

    TI = P(answer_textual) / (P(answer_textual) + P(answer_visual))

so TI > 0.5 is text-biased and TI < 0.5 is image-biased. Note this is the
*normalised* form; the proposal's raw ratio P(text)/P(image) is unbounded and
therefore unusable as a figure axis, so both are computed and the normalised one
is plotted.

**Track B (black-box).** Sampling. Draw N = 8-10 responses at temperature ~0.7;
the Trust Index becomes the empirical fraction of samples matching the textual
answer, and confidence becomes self-consistency (majority-vote agreement),
optionally augmented with a verbalised 0-100 confidence.

Both tracks emit the same downstream columns, which is what lets every figure in
Layer 5 render for every model.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class TrustResult:
    trust_index: float          # in [0, 1]; >0.5 = text-biased
    raw_ratio: float            # the proposal's P(text)/P(image)
    confidence: float           # calibrated logprob (A) or self-consistency (B)
    track: str
    n_samples: int = 1
    p_visual: float = float("nan")
    p_textual: float = float("nan")


# --------------------------------------------------------------------------- #
# Track A
# --------------------------------------------------------------------------- #


def trust_index_track_a(
    adapter,
    *,
    image: str,
    question: str,
    docs: list[str],
    answer_visual: str,
    answer_textual: str,
    meta: dict | None = None,
) -> TrustResult:
    """Constrained scoring of the two candidate answers in the same context."""
    kw = {"meta": meta} if meta is not None else {}
    lp_visual = adapter.score_answer(image, question, docs, answer_visual, **kw)
    lp_textual = adapter.score_answer(image, question, docs, answer_textual, **kw)

    # Softmax over the two length-normalised log-probabilities. Subtracting the
    # max before exponentiating keeps this stable when both are very negative,
    # which they routinely are for multi-token answers.
    m = max(lp_visual, lp_textual)
    p_v = math.exp(lp_visual - m)
    p_t = math.exp(lp_textual - m)
    ti = p_t / (p_t + p_v) if (p_t + p_v) > 0 else 0.5
    ratio = p_t / p_v if p_v > 0 else float("inf")

    return TrustResult(
        trust_index=float(ti),
        raw_ratio=float(ratio),
        confidence=float(max(p_v, p_t) / (p_v + p_t)) if (p_v + p_t) > 0 else 0.5,
        track="A",
        p_visual=float(p_v / (p_v + p_t)) if (p_v + p_t) > 0 else float("nan"),
        p_textual=float(p_t / (p_v + p_t)) if (p_v + p_t) > 0 else float("nan"),
    )


# --------------------------------------------------------------------------- #
# Track B
# --------------------------------------------------------------------------- #


def trust_index_track_b(
    samples: list[str],
    *,
    answer_visual: str,
    answer_textual: str,
    matcher,
) -> TrustResult:
    """Empirical Trust Index from N sampled responses.

    ``matcher(pred, gold) -> bool`` is the same scoring cascade used everywhere
    else, so a Track B trust index and a Track A one are answering the same
    question about the same equivalence classes.

    The index is computed over samples that matched *one of the two* candidates.
    Samples matching neither are off-manifold (refusals, collapse) and would
    otherwise drag the ratio toward 0.5 and mimic balance where there is none;
    their share is recoverable from ``n_samples`` and reported separately.
    """
    n = len(samples)
    if n == 0:
        return TrustResult(float("nan"), float("nan"), float("nan"), "B", 0)

    n_text = sum(1 for s in samples if matcher(s, answer_textual)) if answer_textual else 0
    n_vis = sum(1 for s in samples if matcher(s, answer_visual)) if answer_visual else 0
    denom = n_text + n_vis
    ti = (n_text / denom) if denom else float("nan")
    ratio = (n_text / n_vis) if n_vis else (float("inf") if n_text else float("nan"))

    return TrustResult(
        trust_index=float(ti),
        raw_ratio=float(ratio),
        confidence=self_consistency(samples),
        track="B",
        n_samples=n,
        p_visual=n_vis / n,
        p_textual=n_text / n,
    )


def self_consistency(samples: list[str]) -> float:
    """Majority-vote agreement rate -- Track B's confidence signal.

    Normalised before counting so that "red" and "Red." are not treated as
    disagreement, which would understate confidence for every model.
    """
    from gib.eval.scoring import normalize

    if not samples:
        return float("nan")
    counts = Counter(normalize(s) for s in samples)
    return counts.most_common(1)[0][1] / len(samples)


def majority_answer(samples: list[str]) -> str:
    """The answer scored for accuracy on Track B."""
    from gib.eval.scoring import normalize

    if not samples:
        return ""
    counts = Counter(normalize(s) for s in samples)
    top = counts.most_common(1)[0][0]
    for s in samples:  # return the original surface form, not the normalised one
        if normalize(s) == top:
            return s
    return samples[0]


def parse_verbalized_confidence(text: str) -> float:
    """Extract a self-reported 0-100 confidence, returning NaN when absent."""
    import re

    m = re.search(r"confidence\D{0,10}(\d{1,3})", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d{1,3})\s*%", text)
    if not m:
        return float("nan")
    val = float(m.group(1))
    return val / 100.0 if 0 <= val <= 100 else float("nan")


# --------------------------------------------------------------------------- #
# Counterfactual symmetry
# --------------------------------------------------------------------------- #


def counterfactual_symmetry(ti_a: float, ti_b: float) -> float:
    """Symmetry score for a counterfactual twin pair, in [0, 1].

    1.0 means the model's text bias is identical in both directions -- a
    modality preference. Low values mean the bias is object-dependent, i.e. the
    model is not "trusting text" so much as preferring one particular entity,
    which is a different and weaker claim. This is exactly the question
    counterfactual probing is designed to settle.
    """
    if any(map(lambda x: x is None or np.isnan(x), (ti_a, ti_b))):
        return float("nan")
    return float(1.0 - abs(ti_a - ti_b))
