"""Shared plotting style.

One place for palette and typography so the figure set reads as a single
instrument rather than eight separately-styled plots. The palette is
colour-blind safe (Okabe-Ito derived) and every figure that encodes a category by
colour also encodes it by marker or line style, so the figures survive greyscale
printing -- which is still how a fraction of reviewers read PDFs.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito, reordered so the first few are maximally distinct
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), "-"]

AXIS_COLORS = {
    "clean": "#4C4C4C",
    "noise": "#0072B2",
    "visual": "#D55E00",
    "conflict": "#CC79A7",
    "combined": "#009E73",
}

TAXONOMY_COLORS = {
    "correct": "#009E73",
    "knowledge_inertia": "#D55E00",
    "sensory_capture": "#0072B2",
    "reasoning_collapse": "#7E4AA8",
}

TAXONOMY_LABELS = {
    "correct": "Correct (grounded)",
    "knowledge_inertia": "Knowledge Inertia",
    "sensory_capture": "Sensory Capture",
    "reasoning_collapse": "Reasoning Collapse",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.facecolor": "white",
        }
    )


def model_style(models: list[str]) -> dict[str, dict]:
    """Stable colour/marker/linestyle per model, sorted so a re-run keeps them."""
    return {
        m: {
            "color": PALETTE[i % len(PALETTE)],
            "marker": MARKERS[i % len(MARKERS)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)],
        }
        for i, m in enumerate(sorted(models))
    }


def add_simulation_watermark(fig, enabled: bool = True) -> None:
    """Stamp figures built from simulated adapters.

    Non-optional when simulated rows are present: a figure that leaves the repo
    without this stamp can be mistaken for a result, and that is the single worst
    outcome this codebase could produce.
    """
    if not enabled:
        return
    fig.text(
        0.5, 0.5, "SIMULATED DATA\nNOT A RESULT",
        fontsize=34, color="red", alpha=0.16, ha="center", va="center",
        rotation=28, weight="bold", zorder=1000,
    )


def annotate_thin_cells(ax, flagged: int, total: int) -> None:
    if flagged:
        ax.text(
            0.99, -0.14, f"{flagged}/{total} cells n<30",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#B00020",
        )


def savefig(fig, path, *, simulated: bool = False) -> None:
    from pathlib import Path

    add_simulation_watermark(fig, simulated)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    # PDF alongside PNG: the camera-ready needs vectors.
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
