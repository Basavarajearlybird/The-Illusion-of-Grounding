"""Layer 3 figures -- the internal tug-of-war.

Three publication panels, all rendered from the stored npz captures so they
regenerate without re-running inference:

1. layer-wise VAM curves, clean vs. conflict, with bootstrap confidence bands;
2. the paired-test panel behind "during Knowledge Inertia, visual attention
   collapses";
3. a head-level scatter identifying which heads carry the collapse.

Plus the qualitative side-by-side attention heatmap for one vivid example per
failure mode, which is the figure readers actually remember.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gib.analysis.style import apply_style, savefig
from gib.interp.stats import benjamini_hochberg, layerwise_ci, paired_vam_test


def fig_layerwise_vam(
    clean_vams: dict[str, np.ndarray],
    conflict_vams: dict[str, np.ndarray],
    inertia_vams: dict[str, np.ndarray],
    out: Path,
    model: str = "",
    simulated: bool = False,
) -> dict:
    """Layer-wise VAM curves with 95% bootstrap bands."""
    if not clean_vams or not conflict_vams:
        return {}

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    series = [
        ("Clean (correct)", clean_vams, "#009E73", "-"),
        ("Conflict (all)", conflict_vams, "#0072B2", "--"),
        ("Conflict → Knowledge Inertia", inertia_vams, "#D55E00", "-."),
    ]
    for label, vams, color, ls in series:
        if not vams:
            continue
        mean, lo, hi = layerwise_ci(list(vams.values()))
        if mean.size == 0:
            continue
        layers = np.arange(mean.size)
        ax.plot(layers, mean, color=color, linestyle=ls, label=f"{label}  (n={len(vams)})")
        ax.fill_between(layers, lo, hi, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Visual Attention Mass (VAM)")
    ax.set_title(f"Layer-wise visual attention{' — ' + model if model else ''}")
    ax.legend(loc="best")
    ax.text(0.01, -0.15,
            "VAM = Σ attention to visual-token positions / Σ attention to all positions; "
            "bands are 95% bootstrap CIs",
            transform=ax.transAxes, fontsize=7, color="#555")

    savefig(fig, out / f"fig13_layerwise_vam{'_' + model if model else ''}.png",
            simulated=simulated)
    return {"n_clean": len(clean_vams), "n_conflict": len(conflict_vams),
            "n_inertia": len(inertia_vams)}


def fig_vam_collapse_test(
    matched_clean: dict[str, float],
    matched_inertia: dict[str, float],
    out: Path,
    model: str = "",
    simulated: bool = False,
) -> dict:
    """The headline experiment as a paired test, not a pretty picture.

    Left: per-item paired slopes. Right: the difference distribution with the
    Wilcoxon result and effect size stated on the figure.
    """
    keys = sorted(set(matched_clean) & set(matched_inertia))
    if len(keys) < 6:
        return {"error": f"only {len(keys)} matched pairs; need >= 6"}

    test = paired_vam_test(matched_clean, matched_inertia, keys)
    a = np.array([matched_clean[k] for k in keys])
    b = np.array([matched_inertia[k] for k in keys])

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    for x, y in zip(a, b):
        ax1.plot([0, 1], [x, y], color="#999", alpha=0.28, linewidth=0.7, zorder=1)
    ax1.plot([0, 1], [np.median(a), np.median(b)], color="#D55E00", linewidth=2.6,
             marker="o", zorder=3, label="median")
    ax1.set_xticks([0, 1], ["Clean\n(correct)", "Conflict\n(Knowledge Inertia)"])
    ax1.set_ylabel("Visual Attention Mass")
    ax1.set_title("Paired per-item VAM")
    ax1.legend(fontsize=7.5)

    diffs = b - a
    ax2.hist(diffs, bins=28, color="#0072B2", alpha=0.82)
    ax2.axvline(0, color="#444", linestyle="--", linewidth=1.2)
    ax2.axvline(float(np.median(diffs)), color="#D55E00", linewidth=1.8,
                label=f"median Δ = {np.median(diffs):+.4f}")
    ax2.set_xlabel("Δ VAM  (conflict − clean)")
    ax2.set_ylabel("Items")
    ax2.set_title("Difference distribution")
    ax2.legend(fontsize=7.5)

    fig.suptitle(
        f"Does visual attention collapse during Knowledge Inertia?"
        f"{'  —  ' + model if model else ''}\n"
        f"Wilcoxon signed-rank: p = {test.p_value:.3g},  "
        f"rank-biserial r = {test.effect_size_r:.3f},  n = {test.n_pairs}",
        y=1.04, fontsize=9.5,
    )

    savefig(fig, out / f"fig14_vam_collapse_test{'_' + model if model else ''}.png",
            simulated=simulated)
    return test.as_dict()


def fig_layerwise_significance(
    clean_by_layer: list[np.ndarray],
    inertia_by_layer: list[np.ndarray],
    out: Path,
    model: str = "",
    simulated: bool = False,
) -> dict:
    """Per-layer paired tests with Benjamini-Hochberg FDR correction.

    Testing 32 layers at alpha=.05 without correction is expected to yield ~1.6
    spurious "significant" layers, which a reviewer will spot immediately.
    """
    if not clean_by_layer or not inertia_by_layer:
        return {}

    A = np.vstack(clean_by_layer)     # (items, layers)
    B = np.vstack(inertia_by_layer)
    n_layers = min(A.shape[1], B.shape[1])
    n_items = min(A.shape[0], B.shape[0])
    if n_items < 6:
        return {}

    pvals, diffs = [], []
    for layer in range(n_layers):
        t = paired_vam_test(
            {str(i): float(A[i, layer]) for i in range(n_items)},
            {str(i): float(B[i, layer]) for i in range(n_items)},
        )
        pvals.append(t.p_value)
        diffs.append(t.median_diff)

    sig = benjamini_hochberg(pvals, alpha=0.05)

    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    layers = np.arange(n_layers)
    colors = ["#D55E00" if s else "#BBBBBB" for s in sig]
    ax.bar(layers, diffs, color=colors, alpha=0.9)
    ax.axhline(0, color="#333", linewidth=1.0)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Median Δ VAM (inertia − clean)")
    ax.set_title(f"Where the collapse happens{' — ' + model if model else ''}\n"
                 f"(orange = significant after Benjamini–Hochberg FDR, α=0.05)")

    savefig(fig, out / f"fig15_layerwise_significance{'_' + model if model else ''}.png",
            simulated=simulated)
    return {"n_significant_layers": int(sum(sig)), "n_layers": n_layers}


def fig_head_collapse(
    delta_layer_head: np.ndarray, out: Path, model: str = "", simulated: bool = False
) -> dict:
    """Head-level map of the VAM drop -- which heads carry the collapse."""
    if delta_layer_head is None or getattr(delta_layer_head, "size", 0) == 0:
        return {}

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    vmax = float(np.nanmax(np.abs(delta_layer_head))) or 1e-6
    im = ax1.imshow(delta_layer_head, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("Δ VAM per (layer, head): clean − conflict")
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, shrink=0.85)

    flat = delta_layer_head.ravel()
    flat = flat[~np.isnan(flat)]
    ax2.hist(flat, bins=40, color="#7E4AA8", alpha=0.85)
    ax2.axvline(0, color="#333", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Δ VAM")
    ax2.set_ylabel("Heads")

    top = np.dstack(np.unravel_index(
        np.argsort(-np.nan_to_num(delta_layer_head).ravel())[:5], delta_layer_head.shape
    ))[0]
    ax2.set_title("Distribution across heads\ntop collapsing: " +
                  ", ".join(f"L{layer}H{head}" for layer, head in top), fontsize=8.5)

    savefig(fig, out / f"fig16_head_collapse{'_' + model if model else ''}.png",
            simulated=simulated)
    return {"top_heads": [[int(layer), int(head)] for layer, head in top]}


def fig_attention_example(
    image_path: str,
    grid_clean: np.ndarray | None,
    grid_conflict: np.ndarray | None,
    question: str,
    poisoned_doc: str,
    out: Path,
    name: str = "example",
    simulated: bool = False,
) -> dict:
    """Side-by-side spatial attention for one vivid clean-vs-conflict example.

    Skipped rather than faked when the visual-token count does not reshape to a
    known grid (see ``interp.hooks.visual_attention_grid``): a plausible-looking
    but spatially wrong heatmap is worse than no heatmap.
    """
    if grid_clean is None or grid_conflict is None:
        return {"skipped": "no spatial grid available for this architecture"}

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    apply_style()
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.9))

    axs[0].imshow(img)
    axs[0].set_title("Input image")
    for ax, grid, title in (
        (axs[1], grid_clean, "Clean context\n(image-heavy attention)"),
        (axs[2], grid_conflict, "Conflicting context\n(text-heavy attention)"),
    ):
        ax.imshow(img)
        ax.imshow(
            np.array(Image.fromarray((grid / (grid.max() or 1) * 255).astype(np.uint8))
                     .resize(img.size, Image.BILINEAR)),
            cmap="inferno", alpha=0.55,
        )
        ax.set_title(title)
    for ax in axs:
        ax.axis("off")
        ax.grid(False)

    fig.suptitle(f"Q: {question}", y=1.06, fontsize=9)
    fig.text(0.5, -0.06, f"Poisoned passage: “{poisoned_doc[:180]}”",
             ha="center", fontsize=8, color="#B00020", wrap=True)

    savefig(fig, out / f"fig17_attention_{name}.png", simulated=simulated)
    return {"rendered": True}
