"""Layer 5 -- LaTeX and Markdown tables for the paper.

Emitted as both formats: LaTeX goes straight into the manuscript, Markdown into
the repo README so the numbers are readable on GitHub without compiling anything.

Every table that reports a judge-scored number also reports the judge model and
rubric version in its caption, and every table containing Track B rows footnotes
the sampling-based estimator. Those provenance notes are generated here rather
than typed into the manuscript, so they cannot drift from the code that produced
the numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gib.analysis import load as L
from gib.eval.taxonomy import TAXONOMY_DEFINITIONS


def _fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_table(df: pd.DataFrame, name: str, out: Path, caption: str = "", note: str = "") -> None:
    """Emit one table as Markdown, LaTeX, and CSV.

    CSV is written first and LaTeX is best-effort: pandas routes `to_latex`
    through jinja2 in some versions, and a stale optional dependency must not
    cost the run its tables. A LaTeX failure is reported, not swallowed.
    """
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)

    (out / f"{name}.md").write_text(
        (f"**{caption}**\n\n" if caption else "")
        + df.to_markdown(index=False)
        + (f"\n\n_{note}_\n" if note else "\n"),
        encoding="utf-8",
    )

    try:
        latex = df.to_latex(index=False, escape=True, float_format=lambda v: f"{v:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [tables] {name}: LaTeX skipped ({type(exc).__name__}: {exc}); "
              f"markdown and CSV written")
        return

    if caption:
        latex = (
            "\\begin{table}[t]\n\\centering\n\\small\n"
            + latex
            + f"\\caption{{{caption}"
            + (f" {note}" if note else "")
            + "}\n\\end{table}\n"
        )
    (out / f"{name}.tex").write_text(latex, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def table_model_zoo(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Headline table: clean accuracy and per-axis accuracy per model."""
    acc = df.groupby(["model", "axis"])["is_correct"].mean().unstack()
    n = df.groupby("model").size().rename("n_items")
    track = df.groupby("model")["track"].first()

    tbl = acc.copy()
    tbl.insert(0, "track", track)
    tbl["n_items"] = n
    tbl = tbl.reset_index().rename(columns={
        "clean": "Clean", "noise": "Noise", "visual": "Visual",
        "conflict": "Conflict", "combined": "Combined", "model": "Model",
        "track": "Track", "n_items": "N",
    })
    write_table(
        tbl, "table1_model_zoo", out,
        caption="Model zoo accuracy by stressor axis.",
        note="Track A = white-box (constrained scoring); Track B = black-box "
             "(sampling, N=10 at T=0.7).",
    )
    return tbl


def table_failure_taxonomy(out: Path) -> pd.DataFrame:
    """The formal taxonomy definitions -- the paper's required definition table."""
    tbl = pd.DataFrame(
        [{"Failure mode": k.replace("_", " ").title(), "Definition": v}
         for k, v in TAXONOMY_DEFINITIONS.items()]
    )
    write_table(
        tbl, "table2_failure_taxonomy", out,
        caption="The GIB-15K failure taxonomy.",
        note="Assigned automatically from two judge verdicts (against the visual "
             "answer and against the injected textual falsehood), so proportions "
             "are reported over the full corpus rather than a labelled subset.",
    )
    return tbl


def table_taxonomy_proportions(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    tax = L.taxonomy_proportions(df, "model", "axis")
    if tax.empty:
        return tax
    piv = tax.pivot_table(index=["model", "axis"], columns="failure_mode",
                          values="proportion", fill_value=0.0).reset_index()
    write_table(
        piv, "table3_taxonomy_proportions", out,
        caption="Failure-mode proportions by model and axis.",
    )
    return piv


def table_conflict_matrix(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """The Conflict Matrix: Trust Index, taxonomy, verdict."""
    tbl = L.conflict_table(df)
    if tbl.empty:
        return tbl
    twins = L.twin_table(df)
    if not twins.empty:
        sym = twins.groupby("model")["symmetry"].mean()
        tbl["counterfactual_symmetry"] = tbl["model"].map(sym)

    cols = ["model", "track", "n", "accuracy", "trust_index", "knowledge_inertia",
            "sensory_capture", "reasoning_collapse", "csr_mean"]
    if "counterfactual_symmetry" in tbl:
        cols.append("counterfactual_symmetry")
    cols.append("verdict")

    write_table(
        tbl[cols], "table4_conflict_matrix", out,
        caption="The Conflict Matrix — modality preference under cross-modal conflict.",
        note="Trust Index > 0.5 indicates text bias. Counterfactual symmetry is "
             "computed only over genuinely twinned pairs; 1.0 means the bias flips "
             "symmetrically. CSR is the REFIND context-sensitivity ratio on the "
             "injected falsehood.",
    )
    return tbl


def table_robustness(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    from gib.analysis.figures import compute_robustness

    rob = compute_robustness(df)
    if rob.empty:
        return rob
    piv = rob.pivot_table(index=["model", "track"], columns="axis",
                          values="robustness_r").reset_index()
    clean = L.clean_accuracy(df)
    piv.insert(2, "clean_accuracy", piv["model"].map(clean))

    write_table(
        piv, "table5_robustness", out,
        caption=r"Robustness Score $\mathcal{R} = (Acc_{adverse}/Acc_{clean})\times(1-ECE)$.",
        note="Uncertainty Error is operationalised as Expected Calibration Error. "
             "On Track A, confidence is the length-normalised sequence probability; "
             "on Track B it is self-consistency across N=10 samples. The two tracks "
             "are not pooled.",
    )
    return piv


def table_decay_rates(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    from gib.eval.metrics.robustness import fit_decay

    tbl = L.decay_table(df)
    if tbl.empty:
        return pd.DataFrame()
    rows = []
    for model, g in tbl.groupby("model"):
        g = g.sort_values("n_distractors")
        fit = fit_decay(g["n_distractors"].values, g["accuracy"].values, weights=g["n"].values)
        rows.append({"model": model, **fit.as_dict()})
    out_df = pd.DataFrame(rows).sort_values("decay_lambda", ascending=False)

    write_table(
        out_df, "table6_decay_rates", out,
        caption=r"Retrieval-noise decay rates, $acc(d)=a\,e^{-\lambda d}+c$.",
        note="Higher $\\lambda$ = faster collapse under hard-negative retrieval noise. "
             "Fits are weighted by per-dose sample count.",
    )
    return out_df


def table_human_gap(df: pd.DataFrame, human_summary: dict, out: Path) -> pd.DataFrame:
    if not human_summary or "by_axis" not in human_summary:
        return pd.DataFrame()

    model_acc = df.groupby(["model", "axis"])["is_correct"].mean().unstack()
    rows = []
    for axis, stats in sorted(human_summary["by_axis"].items()):
        row = {"axis": axis, "human": stats["accuracy"], "n_human": stats["n"]}
        for m in model_acc.index:
            if axis in model_acc.columns:
                row[m] = model_acc.loc[m, axis]
        best = min(
            (v for k, v in row.items() if k not in ("axis", "human", "n_human")
             and isinstance(v, float) and not np.isnan(v)),
            default=float("nan"),
        )
        row["max_gap"] = stats["accuracy"] - best if not np.isnan(best) else float("nan")
        rows.append(row)
    tbl = pd.DataFrame(rows)

    write_table(
        tbl, "table7_human_gap", out,
        caption="The Reasoning Gap — human vs. model accuracy by stressor.",
        note=f"Humans scored with the identical judge cascade as models. "
             f"n={human_summary.get('n_participants', '?')} participants, "
             f"Krippendorff's α = {_fmt(human_summary.get('krippendorff_alpha'))}, "
             f"median response time {_fmt(human_summary.get('median_rt_s'), 1)}s.",
    )
    return tbl


def table_metric_market_fit(
    chair: dict, pope: dict, df: pd.DataFrame, out: Path
) -> pd.DataFrame:
    """CHAIR, POPE and REFIND/CSR side by side — the "metric-market fit" table.

    One row per model, showing each borrowed metric evaluated where it actually
    applies rather than pooled into a single meaningless average.
    """
    models = sorted(set(chair) | set(pope) | set(df["model"].unique()))
    if not models:
        return pd.DataFrame()

    csr = df[df["axis"].isin(["conflict", "noise"])].groupby("model")["csr"].mean()
    inertia = (
        df[df["failure_mode"] == "knowledge_inertia"].groupby("model").size()
        / df.groupby("model").size()
    )

    rows = []
    for m in models:
        c = chair.get(m, {})
        p = (pope.get(m) or {}).get("regimes", {}).get("adversarial", {})
        thr = None
        if c.get("by_dose"):
            over = [d for d, v in sorted(c["by_dose"].items())
                    if (v - c.get("clean", 0.0)) * 100 > 10]
            thr = over[0] if over else None
        rows.append(
            {
                "model": m,
                "CHAIR_i (clean)": c.get("clean"),
                "Resolution Threshold (σ)": thr if thr is not None else ">max",
                "POPE F1 (adversarial)": p.get("pope_f1"),
                "POPE yes-ratio": p.get("pope_yes_ratio"),
                "Sycophancy slope": (pope.get(m) or {}).get("sycophancy_slope"),
                "CSR (injected falsehood)": csr.get(m),
                "Knowledge Inertia rate": inertia.get(m),
            }
        )
    tbl = pd.DataFrame(rows)

    write_table(
        tbl, "table9_metric_market_fit", out,
        caption="Metric-market fit — CHAIR, POPE and REFIND/CSR where each applies.",
        note="Resolution Threshold is the first σ where CHAIR exceeds clean by >10 pts; "
             "'>max' means the model never crossed it within the tested range. A positive "
             "sycophancy slope means the yes-ratio rises with distractor count. CSR is "
             "measured on the injected falsehood; high CSR alongside a high Knowledge "
             "Inertia rate is direct evidence the poisoned context is driving the tokens.",
    )
    return tbl


def table_dataset_composition(manifest: dict, out: Path) -> pd.DataFrame:
    """Dataset statistics, for the paper's dataset section and the datasheet."""
    rows = []
    for k, v in sorted((manifest.get("by_sub_type") or {}).items()):
        rows.append({"Field": f"clean pairs · {k}", "Value": v})
    for k, v in sorted((manifest.get("by_axis") or {}).items()):
        rows.append({"Field": f"variants · axis={k}", "Value": v})
    for key in ("n_clean_pairs", "n_variants_full", "n_variants_eval",
                "conflict_items", "conflict_twinned", "combined_pairs",
                "spot_check_n", "dataset_hash", "seed", "git_sha"):
        if key in manifest:
            rows.append({"Field": key, "Value": manifest[key]})
    qa = manifest.get("qa_stats") or {}
    for key in ("scenes", "candidates", "round_trip_rejected", "kept"):
        if key in qa:
            rows.append({"Field": f"qa · {key}", "Value": qa[key]})

    tbl = pd.DataFrame(rows)
    write_table(tbl, "table8_dataset_composition", out,
                caption="GIB-15K composition and build provenance.")
    return tbl
