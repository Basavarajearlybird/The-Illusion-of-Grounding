"""Layer 5 -- result loading and aggregation via DuckDB.

Results are partitioned parquet under ``results/model=*/axis=*/``. DuckDB reads
the hive partitioning directly, so a whole campaign is one query and nothing has
to be materialised in pandas first.

Every aggregate here carries its ``n``. The vulnerability heatmap and the
per-cell taxonomy proportions both have cells that can get thin, and a cell mean
without its n is how a figure ends up asserting something about 3 items.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gib.config import RESULTS_DIR

MIN_CELL_N = 30  # cells below this are rendered but flagged


def _con():
    import duckdb

    return duckdb.connect()


def results_glob(base: Path | None = None) -> str:
    base = Path(base or RESULTS_DIR)
    return str(base / "model=*" / "axis=*" / "results.parquet").replace("\\", "/")


def load_results(base: Path | None = None, models: list[str] | None = None) -> pd.DataFrame:
    """All result rows as a DataFrame."""
    pattern = results_glob(base)
    if not list(Path(base or RESULTS_DIR).glob("model=*/axis=*/results.parquet")):
        raise FileNotFoundError(
            f"no results found under {base or RESULTS_DIR}. Run `make eval MODEL=…` first."
        )
    con = _con()
    df = con.execute(f"SELECT * FROM read_parquet('{pattern}', union_by_name=true)").df()
    if models:
        df = df[df["model"].isin(models)]
    return df


def any_simulated(df: pd.DataFrame) -> bool:
    """True when any row came from a simulated adapter.

    Drives the "SIMULATED DATA" watermark on every figure. Checked on the data
    rather than on a flag passed by the caller, so a mixed campaign cannot lose
    the warning.
    """
    if "simulated" in df.columns:
        return bool(df["simulated"].any())
    return bool(df["model"].astype(str).str.startswith("sim-").any())


def accuracy_by(df: pd.DataFrame, *keys: str) -> pd.DataFrame:
    """Grouped accuracy with n, taxonomy shares, and mean confidence."""
    g = df.groupby(list(keys), dropna=False)
    out = g.agg(
        accuracy=("is_correct", "mean"),
        n=("is_correct", "size"),
        confidence=("confidence", "mean"),
        trust_index=("trust_index", "mean"),
        csr=("csr", "mean"),
        latency_s=("latency_s", "mean"),
    ).reset_index()
    out["thin_cell"] = out["n"] < MIN_CELL_N
    return out


def clean_accuracy(df: pd.DataFrame) -> pd.Series:
    """Per-model clean accuracy -- the denominator for every ratio in this layer."""
    clean = df[df["axis"] == "clean"]
    return clean.groupby("model")["is_correct"].mean()


def taxonomy_proportions(df: pd.DataFrame, *keys: str) -> pd.DataFrame:
    """Failure-mode shares within each group."""
    keys = list(keys) or ["model"]
    counts = (
        df.groupby(keys + ["failure_mode"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby(keys)["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts


def decay_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy vs. distractor count, per model -- input to the decay fit."""
    noise = df[df["axis"] == "noise"].copy()
    noise["n_distractors"] = pd.to_numeric(noise["n_distractors"], errors="coerce")
    out = (
        noise.groupby(["model", "n_distractors"])
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "size"))
        .reset_index()
    )
    # Dose 0 is the clean condition; the decay fit needs the intercept and the
    # noise axis by construction never contains a zero-distractor row.
    clean = clean_accuracy(df)
    zeros = pd.DataFrame(
        {
            "model": clean.index,
            "n_distractors": 0.0,
            "accuracy": clean.values,
            "n": df[df["axis"] == "clean"].groupby("model").size().reindex(clean.index).values,
        }
    )
    return pd.concat([zeros, out], ignore_index=True).sort_values(["model", "n_distractors"])


def visual_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy vs. degradation dose, per model and degradation family."""
    vis = df[df["axis"] == "visual"].copy()
    return (
        vis.groupby(["model", "degradation", "degradation_param"])
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "size"),
             confidence=("confidence", "mean"))
        .reset_index()
        .sort_values(["model", "degradation", "degradation_param"])
    )


def heatmap_table(df: pd.DataFrame) -> pd.DataFrame:
    """The combined-stressor grid: blur (y) x distractors (x)."""
    comb = df[df["axis"] == "combined"].copy()
    comb["degradation_param"] = pd.to_numeric(comb["degradation_param"], errors="coerce")
    comb["n_distractors"] = pd.to_numeric(comb["n_distractors"], errors="coerce")
    grid = (
        comb.groupby(["model", "degradation_param", "n_distractors"])
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "size"))
        .reset_index()
    )
    clean = clean_accuracy(df)
    grid["acc_clean"] = grid["model"].map(clean)
    grid["delta"] = grid["accuracy"] - grid["acc_clean"]
    grid["thin_cell"] = grid["n"] < MIN_CELL_N
    return grid


def conflict_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model conflict summary: Trust Index, taxonomy shares, verdict."""
    conf = df[df["axis"] == "conflict"]
    if conf.empty:
        return pd.DataFrame()

    rows = []
    for model, g in conf.groupby("model"):
        ti = g["trust_index"].mean(skipna=True)
        tax = g["failure_mode"].value_counts(normalize=True)
        rows.append(
            {
                "model": model,
                "track": g["track"].iloc[0],
                "n": len(g),
                "trust_index": ti,
                "accuracy": g["is_correct"].mean(),
                "knowledge_inertia": tax.get("knowledge_inertia", 0.0),
                "sensory_capture": tax.get("sensory_capture", 0.0),
                "reasoning_collapse": tax.get("reasoning_collapse", 0.0),
                "csr_mean": g["csr"].mean(skipna=True),
                # The verdict column the paper's Conflict Matrix reports.
                "verdict": (
                    "Text-Biased" if ti > 0.55
                    else "Image-Biased" if ti < 0.45
                    else "Balanced"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("trust_index", ascending=False)


def twin_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counterfactual symmetry, computed only over genuinely twinned pairs."""
    conf = df[(df["axis"] == "conflict") & (df["twin_id"].astype(str) != "")]
    if conf.empty:
        return pd.DataFrame()

    rows = []
    for (model, twin), g in conf.groupby(["model", "twin_id"]):
        if len(g) != 2:
            continue  # an unmatched half is not a counterfactual pair
        a, b = g.iloc[0], g.iloc[1]
        rows.append(
            {
                "model": model, "twin_id": twin,
                "ti_a": a["trust_index"], "ti_b": b["trust_index"],
                "symmetry": 1.0 - abs(a["trust_index"] - b["trust_index"]),
                "both_text": bool(a["matches_textual"] and b["matches_textual"]),
                "both_visual": bool(a["is_correct"] and b["is_correct"]),
            }
        )
    return pd.DataFrame(rows)


def position_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by the ground-truth document's position -- the free ablation.

    The dataset logs `gt_doc_position` for every noise row, so this costs nothing
    beyond a groupby and answers a question reviewers reliably ask.
    """
    noise = df[(df["axis"] == "noise") & (df["gt_doc_position"] >= 0)]
    if noise.empty:
        return pd.DataFrame()
    return (
        noise.groupby(["model", "gt_doc_position"])
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "size"))
        .reset_index()
    )
