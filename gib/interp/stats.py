"""Statistics for the interpretability claims.

The paper's claim -- "during Knowledge Inertia, visual attention collapses" --
is a statistical assertion, so it is tested as one. A figure showing two
distributions that look different is not evidence a reviewer will accept when the
comparison is available as a paired test on matched items.

Wilcoxon signed-rank rather than a paired t-test: VAM is a bounded ratio, its
distribution is skewed, and per-layer samples are not close to normal. The
matched-pairs rank test makes no distributional assumption and is the standard
choice here.

Effect size is reported alongside p, because n is in the hundreds and a p-value
alone will be significant for effects too small to matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedTest:
    statistic: float
    p_value: float
    n_pairs: int
    median_a: float
    median_b: float
    median_diff: float
    effect_size_r: float       # rank-biserial correlation
    cliffs_delta: float

    def as_dict(self) -> dict:
        return {
            "wilcoxon_stat": round(self.statistic, 4),
            "p_value": float(f"{self.p_value:.3g}") if self.p_value == self.p_value else float("nan"),
            "n_pairs": self.n_pairs,
            "median_a": round(self.median_a, 4),
            "median_b": round(self.median_b, 4),
            "median_diff": round(self.median_diff, 4),
            "effect_size_r": round(self.effect_size_r, 4),
            "cliffs_delta": round(self.cliffs_delta, 4),
        }

    def summary(self, name_a: str = "A", name_b: str = "B") -> str:
        return (
            f"{name_a} median {self.median_a:.4f} vs {name_b} median {self.median_b:.4f} "
            f"(diff {self.median_diff:+.4f}, n={self.n_pairs}, "
            f"Wilcoxon p={self.p_value:.3g}, r={self.effect_size_r:.3f})"
        )


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]; 0 means complete overlap."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Rank-based computation: O(n log n) instead of the naive O(n*m) all-pairs.
    combined = np.concatenate([a, b])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]

    n1, n2 = a.size, b.size
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(2.0 * u1 / (n1 * n2) - 1.0)


def paired_vam_test(vam_a: dict[str, float], vam_b: dict[str, float], keys=None) -> PairedTest:
    """Paired Wilcoxon signed-rank over matched items.

    ``vam_a`` and ``vam_b`` are ``{match_key: vam}``; only keys present in both
    contribute, which is what keeps the test paired.
    """
    from scipy.stats import wilcoxon

    keys = sorted(set(vam_a) & set(vam_b)) if keys is None else list(keys)
    a = np.array([vam_a[k] for k in keys], dtype=float)
    b = np.array([vam_b[k] for k in keys], dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]

    if a.size < 6:
        return PairedTest(float("nan"), float("nan"), int(a.size),
                          float(np.median(a)) if a.size else float("nan"),
                          float(np.median(b)) if b.size else float("nan"),
                          float("nan"), float("nan"), float("nan"))

    diffs = a - b
    if np.allclose(diffs, 0):
        return PairedTest(0.0, 1.0, int(a.size), float(np.median(a)), float(np.median(b)),
                          0.0, 0.0, 0.0)

    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")

    # Rank-biserial correlation: (W+ - W-) / (W+ + W-)
    nz = diffs[diffs != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    r = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) else float("nan")

    return PairedTest(
        statistic=float(stat), p_value=float(p), n_pairs=int(a.size),
        median_a=float(np.median(a)), median_b=float(np.median(b)),
        median_diff=float(np.median(diffs)),
        effect_size_r=float(r), cliffs_delta=cliffs_delta(a, b),
    )


def layerwise_ci(vams: list[np.ndarray], confidence: float = 0.95):
    """Per-layer mean and bootstrap confidence band for the VAM curves."""
    if not vams:
        return np.array([]), np.array([]), np.array([])
    arr = np.vstack([np.asarray(v, dtype=float) for v in vams])  # (items, layers)
    mean = np.nanmean(arr, axis=0)

    rng = np.random.default_rng(20260721)
    n = arr.shape[0]
    boots = np.stack(
        [np.nanmean(arr[rng.integers(0, n, n)], axis=0) for _ in range(1000)]
    )
    lo = np.nanpercentile(boots, (1 - confidence) / 2 * 100, axis=0)
    hi = np.nanpercentile(boots, (1 + confidence) / 2 * 100, axis=0)
    return mean, lo, hi


def head_level_collapse(
    heads_clean: list[np.ndarray], heads_conflict: list[np.ndarray]
) -> np.ndarray:
    """Per-(layer, head) VAM drop from clean to conflict.

    Identifies which heads carry the collapse. Recent work on attention-head
    imbalance under modality conflict makes this a well-timed and citable angle,
    and it is cheap once the captures exist.
    """
    if not heads_clean or not heads_conflict:
        return np.array([])
    c = np.nanmean(np.stack(heads_clean), axis=0)
    f = np.nanmean(np.stack(heads_conflict), axis=0)
    return c - f


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """FDR correction.

    Applied to the per-layer tests: with 32 layers, testing each one at alpha=.05
    would be expected to produce roughly 1.6 spurious "significant" layers, and a
    reviewer will notice.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresholds
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out.tolist()
