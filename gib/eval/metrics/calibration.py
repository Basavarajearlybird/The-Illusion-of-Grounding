"""Calibration -- the operationalisation of the Robustness Score's
"Uncertainty Error" term.

The proposal defines

    R = (Acc_adverse / Acc_clean) x (1 - Uncertainty Error)

but leaves "Uncertainty Error" unspecified. The architecture operationalises it
as **Expected Calibration Error**, and the choice is exactly right for what the
score is meant to penalise: R is designed to punish models that are *confident
but wrong*, and that is what high ECE literally is. A model that degrades under
noise while correctly becoming less certain is penalised only through the
accuracy ratio; a model that degrades while staying certain is penalised twice.

Both tracks feed the same estimator:

* Track A -- confidence is the calibrated probability from token logprobs.
* Track B -- confidence is self-consistency across N samples (optionally
  averaged with a verbalised 0-100 confidence).

The tracks are *not* pooled into a single ECE. They are different instruments,
and averaging them would produce a number belonging to neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    ece: float
    mce: float                       # maximum calibration error
    brier: float
    n: int
    n_bins: int
    bin_edges: list[float] = field(default_factory=list)
    bin_acc: list[float] = field(default_factory=list)
    bin_conf: list[float] = field(default_factory=list)
    bin_count: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier": round(self.brier, 4),
            "n": self.n,
        }


def expected_calibration_error(
    confidences, correctness, n_bins: int = 10
) -> CalibrationResult:
    """Equal-width binned ECE (Guo et al., 2017).

    Rows with NaN confidence are dropped rather than imputed -- a model that
    exposes no confidence signal for an item has no calibration to measure there,
    and imputing 0.5 would manufacture calibration error out of missing data.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    mask = ~np.isnan(conf) & ~np.isnan(corr)
    conf, corr = conf[mask], corr[mask]
    n = len(conf)
    if n == 0:
        return CalibrationResult(float("nan"), float("nan"), float("nan"), 0, n_bins)

    conf = np.clip(conf, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive on the last bin so confidence == 1.0 is not dropped
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    bin_acc, bin_conf, bin_count = [], [], []
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        bin_count.append(count)
        if count == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
            continue
        acc_b = float(corr[sel].mean())
        conf_b = float(conf[sel].mean())
        bin_acc.append(acc_b)
        bin_conf.append(conf_b)
        gap = abs(acc_b - conf_b)
        ece += (count / n) * gap
        mce = max(mce, gap)

    brier = float(np.mean((conf - corr) ** 2))
    return CalibrationResult(
        ece=float(ece), mce=float(mce), brier=brier, n=n, n_bins=n_bins,
        bin_edges=[float(e) for e in edges], bin_acc=bin_acc,
        bin_conf=bin_conf, bin_count=bin_count,
    )


def confidence_from_logprobs(token_logprobs: list[float] | None) -> float:
    """Track A confidence: geometric mean token probability.

    Geometric rather than arithmetic mean, i.e. the length-normalised sequence
    probability. An arithmetic mean over probabilities would let one high-
    probability function word mask a low-probability content word, which is
    exactly the token whose confidence matters.
    """
    if not token_logprobs:
        return float("nan")
    return float(np.exp(np.mean(token_logprobs)))


def overconfidence_rate(confidences, correctness, threshold: float = 0.8) -> float:
    """Share of items answered wrong at confidence above ``threshold``.

    The single most quotable calibration number for this paper: "model X is
    >80% confident on N% of the items it gets wrong" states the Illusion of
    Grounding in one sentence.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    mask = ~np.isnan(conf) & ~np.isnan(corr)
    conf, corr = conf[mask], corr[mask]
    if len(conf) == 0:
        return float("nan")
    wrong = corr < 0.5
    if wrong.sum() == 0:
        return 0.0
    return float((conf[wrong] > threshold).mean())
