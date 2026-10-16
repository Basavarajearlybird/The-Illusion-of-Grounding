"""The Robustness Score R, and the decay-rate fit.

    R = (Accuracy_adverse / Accuracy_clean) x (1 - ECE)

Reported per axis and pooled, with the Track A/B uncertainty operationalisation
footnoted, because the two tracks estimate the ECE term with different
instruments (see :mod:`gib.eval.metrics.calibration`).

Two properties of this formula are worth stating in the paper because reviewers
will probe them:

* R is a *ratio*, so a model with low clean accuracy can score well by degrading
  gracefully. That is intended -- R measures robustness, not capability -- but it
  means R must always be reported *alongside* clean accuracy, never instead of
  it. That pairing is exactly what the Pareto frontier plots.
* R is bounded above by 1 only when adverse accuracy cannot exceed clean
  accuracy. It occasionally can, by noise, on small cells; the value is not
  clipped, because a clipped R would hide a cell whose n is too small to trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RobustnessResult:
    r: float
    acc_clean: float
    acc_adverse: float
    ratio: float
    ece: float
    n_clean: int
    n_adverse: int
    track: str = ""

    def as_dict(self) -> dict:
        return {
            "robustness_r": round(self.r, 4),
            "acc_clean": round(self.acc_clean, 4),
            "acc_adverse": round(self.acc_adverse, 4),
            "acc_ratio": round(self.ratio, 4),
            "ece": round(self.ece, 4),
            "n_clean": self.n_clean,
            "n_adverse": self.n_adverse,
            "track": self.track,
        }


def robustness_score(
    acc_clean: float, acc_adverse: float, ece: float,
    n_clean: int = 0, n_adverse: int = 0, track: str = "",
) -> RobustnessResult:
    ratio = acc_adverse / acc_clean if acc_clean > 0 else float("nan")
    # A missing ECE must not silently become a perfect one: without a confidence
    # signal the penalty term is undefined, so R is undefined too.
    r = ratio * (1.0 - ece) if not math.isnan(ece) else float("nan")
    return RobustnessResult(
        r=float(r), acc_clean=float(acc_clean), acc_adverse=float(acc_adverse),
        ratio=float(ratio), ece=float(ece), n_clean=n_clean, n_adverse=n_adverse,
        track=track,
    )


# --------------------------------------------------------------------------- #
# Decay rate (Axis 1's elite metric)
# --------------------------------------------------------------------------- #


@dataclass
class DecayFit:
    """Fit of ``acc(d) = a * exp(-lambda * d) + c``."""

    lam: float          # the decay rate -- one quotable scalar per model
    a: float
    c: float
    r_squared: float
    half_life: float    # distractors to lose half the decayable accuracy
    n_points: int
    converged: bool = True

    def as_dict(self) -> dict:
        return {
            "decay_lambda": round(self.lam, 5),
            "decay_a": round(self.a, 4),
            "decay_floor": round(self.c, 4),
            "decay_r2": round(self.r_squared, 4),
            "decay_half_life": round(self.half_life, 3),
            "decay_n_points": self.n_points,
            "decay_converged": self.converged,
        }

    def predict(self, d):
        d = np.asarray(d, dtype=float)
        return self.a * np.exp(-self.lam * d) + self.c


def fit_decay(doses, accuracies, *, weights=None) -> DecayFit:
    """Fit the exponential decay and report lambda as the model's decay rate.

    A single scalar per model is far more quotable than a curve, and it is what
    makes models rankable on Axis 1.

    ``weights`` should be the per-dose sample counts. Without them the dose with
    n=50 pulls the fit exactly as hard as the dose with n=5000, which is how a
    thin cell ends up dictating a headline number.
    """
    from scipy.optimize import curve_fit

    d = np.asarray(doses, dtype=float)
    y = np.asarray(accuracies, dtype=float)
    mask = ~np.isnan(y)
    d, y = d[mask], y[mask]
    if weights is not None:
        w = np.asarray(weights, dtype=float)[mask]
        sigma = 1.0 / np.sqrt(np.maximum(w, 1.0))
    else:
        sigma = None

    if len(d) < 3:
        return DecayFit(float("nan"), float("nan"), float("nan"), float("nan"),
                        float("nan"), len(d), converged=False)

    def model(x, a, lam, c):
        return a * np.exp(-lam * x) + c

    p0 = [max(1e-3, y[0] - y[-1]), 0.1, max(0.0, float(y.min()))]
    try:
        popt, _ = curve_fit(
            model, d, y, p0=p0, sigma=sigma, maxfev=20000,
            bounds=([0.0, 1e-4, 0.0], [1.5, 5.0, 1.0]),
        )
        converged = True
    except Exception:
        popt, converged = p0, False

    a, lam, c = [float(v) for v in popt]
    resid = y - model(d, a, lam, c)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    half_life = math.log(2) / lam if lam > 0 else float("inf")

    return DecayFit(lam, a, c, r2, half_life, len(d), converged)


def relative_drop(acc_clean: float, acc_adverse: float) -> float:
    """Delta-accuracy against clean, as a fraction -- the heatmap's cell value."""
    if acc_clean <= 0:
        return float("nan")
    return float((acc_adverse - acc_clean) / acc_clean)
