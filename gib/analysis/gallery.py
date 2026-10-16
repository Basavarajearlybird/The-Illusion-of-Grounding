"""The Hallucination Gallery -- auto-mined qualitative failures.

Query for high-confidence Knowledge Inertia failures shared across >= 3 models,
then render image + poisoned passage + every model's output as a figure grid.
Citable, memorable, and cheap because it is mined from results that already exist
rather than curated by hand.

The cross-model requirement is what makes an example worth printing: a failure
one model makes is an anecdote, and a failure every model makes on the same item
is a property of the item. The confidence filter then picks the ones that
illustrate the Illusion of Grounding most sharply -- fluent, certain, and wrong.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gib.analysis.style import apply_style, savefig
from gib.schema import FailureMode


def mine_gallery(
    df: pd.DataFrame,
    variants: list[dict],
    *,
    min_models: int = 3,
    min_confidence: float = 0.7,
    top_k: int = 12,
    failure_mode: str = FailureMode.KNOWLEDGE_INERTIA.value,
) -> list[dict]:
    """Find the items that fooled the most models, most confidently."""
    fails = df[df["failure_mode"] == failure_mode]
    if fails.empty:
        return []

    by_variant = {v["variant_id"]: v for v in variants}
    rows = []

    for vid, g in fails.groupby("variant_id"):
        models = sorted(g["model"].unique())
        if len(models) < min_models:
            continue
        conf = g["confidence"].mean(skipna=True)
        if pd.notna(conf) and conf < min_confidence:
            continue
        var = by_variant.get(vid)
        if var is None:
            continue
        rows.append(
            {
                "variant_id": vid,
                "n_models_fooled": len(models),
                "mean_confidence": float(conf) if pd.notna(conf) else float("nan"),
                "axis": g["axis"].iloc[0],
                "dose": g["dose"].iloc[0],
                "sub_type": g["sub_type"].iloc[0],
                "question": var["question"],
                "answer_visual": var["answer_visual"],
                "answer_textual": var["answer_textual"],
                "image_path": var["image_path"],
                "poisoned_docs": list(var["retrieved_docs"] or []),
                "model_answers": {r["model"]: r["answer"] for _, r in g.iterrows()},
            }
        )

    rows.sort(key=lambda r: (-r["n_models_fooled"], -(r["mean_confidence"] or 0)))
    return rows[:top_k]


def render_gallery(
    entries: list[dict], out: Path, *, n_show: int = 6, simulated: bool = False
) -> dict:
    """Render the gallery as a figure grid."""
    entries = entries[:n_show]
    if not entries:
        return {}

    apply_style()
    ncols = 2
    nrows = int((len(entries) + ncols - 1) / ncols)
    fig, axs = plt.subplots(nrows, ncols * 2, figsize=(11.0, 3.5 * nrows),
                            gridspec_kw={"width_ratios": [1, 1.35] * ncols},
                            squeeze=False)

    from PIL import Image

    for i, entry in enumerate(entries):
        r, c = divmod(i, ncols)
        ax_img = axs[r][c * 2]
        ax_txt = axs[r][c * 2 + 1]

        try:
            ax_img.imshow(Image.open(entry["image_path"]).convert("RGB"))
        except Exception:
            ax_img.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        ax_img.axis("off")
        ax_img.set_title(f"{entry['sub_type']} · {entry['axis']}", fontsize=8)

        doc = entry["poisoned_docs"][0] if entry["poisoned_docs"] else ""
        body = (
            f"Q: {entry['question']}\n\n"
            f"TRUTH (image): {entry['answer_visual']}\n"
            f"POISON (text): {entry['answer_textual']}\n\n"
            f"Passage: “{textwrap.shorten(doc, 190, placeholder='…')}”\n\n"
            "Model answers:\n"
            + "\n".join(f"  · {m}: {a}" for m, a in sorted(entry["model_answers"].items()))
            + f"\n\n{entry['n_models_fooled']} models fooled"
            + (f", mean conf {entry['mean_confidence']:.2f}"
               if entry["mean_confidence"] == entry["mean_confidence"] else "")
        )
        ax_txt.text(0.0, 1.0, body, va="top", ha="left", fontsize=7.0, family="monospace",
                    transform=ax_txt.transAxes, wrap=True)
        ax_txt.axis("off")

    for j in range(len(entries), nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r][c * 2].axis("off")
        axs[r][c * 2 + 1].axis("off")

    fig.suptitle("The Hallucination Gallery — high-confidence Knowledge Inertia "
                 "failures shared across models", y=1.01)

    savefig(fig, out / "fig18_hallucination_gallery.png", simulated=simulated)
    return {"n_entries": len(entries)}


def save_gallery_json(entries: list[dict], path: Path) -> Path:
    """Machine-readable gallery, released alongside the dataset.

    Downstream users get the mined examples without having to re-run the zoo,
    which is most of the gallery's citation value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path
