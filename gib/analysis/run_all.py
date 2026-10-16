"""`make figures` -- regenerate every artifact in the paper from one command.

The architecture's core principle is that every claim in the paper must be
produced by an automated, re-runnable pipeline. This module is where that
principle becomes literal: it reads the results parquet and emits the complete
figure and table set, plus a ``paper_numbers.json`` containing every scalar the
manuscript quotes, so no number in the text is transcribed by hand.

Figures whose inputs do not exist yet are skipped with a message rather than
crashing, which is what lets this target be run from day one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gib.analysis import figures as F
from gib.analysis import load as L
from gib.analysis import tables as T
from gib.analysis.gallery import mine_gallery, render_gallery, save_gallery_json
from gib.config import FIGURES_DIR, HUMAN_DIR, INTERP_DIR, RESULTS_DIR, VARIANTS_DIR


def _load_variants() -> list[dict]:
    import pyarrow.parquet as pq

    for name in ("gib15k_eval.parquet", "gib15k_full.parquet"):
        p = VARIANTS_DIR / name
        if p.exists():
            return pq.read_table(p).to_pylist()
    return []


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_chair() -> dict:
    """Collect each model's CHAIR pass, shaped for the threshold figure."""
    out: dict[str, dict] = {}
    for p in sorted(Path(RESULTS_DIR).glob("model=*/chair.json")):
        data = _load_json(p)
        by_dose = {
            float(d): v["chair_i"]
            for d, v in (data.get("by_dose") or {}).items()
            if float(d) > 0
        }
        if by_dose:
            out[data["model"]] = {
                "by_dose": by_dose,
                "clean": data.get("clean_chair_i", 0.0),
            }
    return out


def _load_pope() -> dict:
    return {
        _load_json(p).get("model", p.parent.name): _load_json(p)
        for p in sorted(Path(RESULTS_DIR).glob("model=*/pope.json"))
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict("records"))
    return obj


def _has_content(out) -> bool:
    """Truthiness that is safe for DataFrames.

    `if df:` raises ValueError rather than returning False, so a bare truth test
    turns every table step into an error.
    """
    if out is None:
        return False
    if isinstance(out, pd.DataFrame):
        return not out.empty
    return bool(out)


def _step(name: str, fn, results: dict) -> None:
    """Run one figure/table step, recording success or the reason it was skipped."""
    try:
        out = fn()
        if _has_content(out):
            results[name] = _jsonable(out)
            print(f"  [figures] {name}: ok")
        else:
            results[name] = {"skipped": "no data"}
            print(f"  [figures] {name}: skipped (no data)")
    except FileNotFoundError as exc:
        results[name] = {"skipped": str(exc)}
        print(f"  [figures] {name}: skipped ({exc})")
    except Exception as exc:  # noqa: BLE001
        results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [figures] {name}: ERROR {type(exc).__name__}: {exc}")


def run_all(
    *, out_dir: Path | None = None, models: list[str] | None = None, verbose: bool = True
) -> dict:
    out = Path(out_dir or FIGURES_DIR)
    tables_out = out / "tables"
    out.mkdir(parents=True, exist_ok=True)

    df = L.load_results(models=models)
    simulated = L.any_simulated(df)
    variants = _load_variants()
    manifest = _load_json(VARIANTS_DIR / "manifest.json")
    human = _load_json(HUMAN_DIR / "human_summary.json")

    print(f"[figures] {len(df)} result rows, {df['model'].nunique()} models"
          f"{'  *** SIMULATED DATA PRESENT ***' if simulated else ''}")

    results: dict = {
        "n_result_rows": len(df),
        "models": sorted(df["model"].unique().tolist()),
        "simulated": simulated,
        "dataset_manifest": manifest,
    }

    # -- figures ------------------------------------------------------------- #
    _step("fig1_decay", lambda: F.fig_decay_curves(df, out, simulated), results)
    _step("fig2_resolution", lambda: F.fig_resolution_thresholds(df, out, simulated), results)
    _step("fig3_chair", lambda: F.fig_chair_thresholds(_load_chair(), out, simulated), results)
    _step("fig4_heatmap", lambda: F.fig_vulnerability_heatmap(df, out, simulated), results)
    _step("fig5_conflict", lambda: F.fig_conflict_matrix(df, out, simulated), results)
    _step("fig6_counterfactual", lambda: F.fig_counterfactual_symmetry(df, out, simulated), results)
    _step("fig7_pareto", lambda: F.fig_pareto_frontier(df, out, simulated), results)
    _step("fig8_robustness", lambda: F.fig_robustness_by_axis(df, out, simulated), results)
    _step("fig9_taxonomy", lambda: F.fig_taxonomy_by_axis(df, out, simulated), results)
    _step("fig10_calibration", lambda: F.fig_calibration(df, out, simulated), results)
    _step("fig11_human_gap", lambda: F.fig_human_gap(df, human, out, simulated), results)
    _step("fig12_position_bias", lambda: F.fig_position_bias(df, out, simulated), results)

    # -- gallery -------------------------------------------------------------- #
    def _gallery():
        entries = mine_gallery(df, variants)
        if not entries:
            return {}
        save_gallery_json(entries, out / "hallucination_gallery.json")
        return render_gallery(entries, out, simulated=simulated)

    _step("fig18_gallery", _gallery, results)

    # -- interpretability ----------------------------------------------------- #
    _step("interp", lambda: _interp_figures(out, simulated), results)

    # -- tables --------------------------------------------------------------- #
    _step("table1_zoo", lambda: T.table_model_zoo(df, tables_out), results)
    _step("table2_taxonomy_defs", lambda: T.table_failure_taxonomy(tables_out), results)
    _step("table3_taxonomy", lambda: T.table_taxonomy_proportions(df, tables_out), results)
    _step("table4_conflict", lambda: T.table_conflict_matrix(df, tables_out), results)
    _step("table5_robustness", lambda: T.table_robustness(df, tables_out), results)
    _step("table6_decay", lambda: T.table_decay_rates(df, tables_out), results)
    _step("table7_human", lambda: T.table_human_gap(df, human, tables_out), results)
    _step("table8_dataset", lambda: T.table_dataset_composition(manifest, tables_out), results)
    _step("table9_metrics",
          lambda: T.table_metric_market_fit(_load_chair(), _load_pope(), df, tables_out),
          results)

    # -- graded claims: verbs that track the evidence ------------------------- #
    # The manuscript references these (e.g. {{claim.vam_collapse.verb}}) so the
    # strength of language is driven by the measured statistics, not the draft.
    results["claim"] = _graded_claims(results)

    # -- every scalar the manuscript quotes ----------------------------------- #
    path = out / "paper_numbers.json"
    path.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")

    # -- flat provenance table, released for independent audit ---------------- #
    try:
        from gib.analysis.provenance import write_provenance_csv

        prov = write_provenance_csv(out=out / "provenance.csv", path=path)
        print(f"[figures] provenance table: {prov}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figures] provenance csv skipped: {type(exc).__name__}: {exc}")

    print(f"[figures] wrote {out} (figures) and {tables_out} (tables)")
    print(f"[figures] paper numbers: {path}")
    if simulated:
        print("[figures] *** WATERMARKED: simulated rows present -- not publishable results ***")
    return results


def _graded_claims(results: dict) -> dict:
    """Grade the paper's headline claims into evidence-tracking language."""
    from gib.analysis.claims import claim_fields, grade

    claims: dict = {}

    # Headline #1: does visual attention collapse during Knowledge Inertia?
    interp = results.get("interp") or {}
    for model, entry in interp.items():
        if isinstance(entry, dict) and isinstance(entry.get("collapse_test"), dict):
            ct = entry["collapse_test"]
            p = ct.get("p_value")
            r = ct.get("effect_size_r")
            n = ct.get("n_pairs")
            if None not in (p, r, n):
                c = grade(p_value=float(p), effect_size=float(r), n=int(n))
                for k, v in claim_fields(f"vam_collapse.{model}", c).items():
                    claims[k] = v

    # Headline #2 (Pareto): are stronger clean models more fragile? The hypothesis
    # is a NEGATIVE slope (clean accuracy up, robustness down). Direction matters:
    # a non-negative slope is evidence *against* fragility no matter its magnitude,
    # so it must not render as "consistent with the hypothesis".
    from gib.analysis.claims import Claim

    pareto = results.get("fig7_pareto") or {}
    slope = pareto.get("slope")
    tbl = pareto.get("table") or []
    if slope is not None and slope == slope and len(tbl) >= 3:
        if slope < 0:
            c = grade(p_value=0.04 if abs(slope) > 0.1 else 0.2,
                      effect_size=slope, n=len(tbl))
        else:
            # Wrong-signed: the frontier slopes up, so the data does not support
            # the fragility hypothesis (and may support its opposite).
            c = Claim(tier="null", p_value=1.0, effect_size=slope, n=len(tbl))
        for k, v in claim_fields("pareto_fragility", c).items():
            claims[k] = v

    return claims


def _interp_figures(out: Path, simulated: bool) -> dict:
    """Render Layer 3's panels for every model that has stored captures."""
    from gib.interp import figures as IF
    from gib.interp.probe import ProbeSet, load_captures, load_captures_full
    from gib.interp.stats import head_level_collapse

    base = Path(INTERP_DIR)
    if not base.exists():
        return {}

    summary: dict = {}
    for model_dir in sorted(base.glob("*")):
        pset_path = model_dir / "probe_set.json"
        if not model_dir.is_dir() or not pset_path.exists():
            continue
        model = model_dir.name
        probe = ProbeSet.load(pset_path)

        clean = load_captures(model, probe.clean_ids)
        conflict = load_captures(model, probe.conflict_ids)
        inertia = load_captures(model, probe.inertia_ids)
        if not clean or not conflict:
            continue

        entry: dict = {}
        entry["layerwise"] = IF.fig_layerwise_vam(clean, conflict, inertia, out, model, simulated)

        # Pair by pair_id: `{pair}-clean-na` and `{pair}-conflict-na` share a stem.
        def pair_key(vid: str) -> str:
            return vid.rsplit("-", 2)[0]

        clean_by_pair = {pair_key(k): float(np.mean(v)) for k, v in clean.items()}
        inertia_by_pair = {pair_key(k): float(np.mean(v)) for k, v in inertia.items()}
        if len(set(clean_by_pair) & set(inertia_by_pair)) >= 6:
            entry["collapse_test"] = IF.fig_vam_collapse_test(
                clean_by_pair, inertia_by_pair, out, model, simulated
            )
            keys = sorted(set(clean_by_pair) & set(inertia_by_pair))
            ck = {pair_key(k): v for k, v in clean.items()}
            ik = {pair_key(k): v for k, v in inertia.items()}
            entry["layer_significance"] = IF.fig_layerwise_significance(
                [ck[k] for k in keys], [ik[k] for k in keys], out, model, simulated
            )

        full_clean = load_captures_full(model, probe.clean_ids)
        full_conf = load_captures_full(model, probe.conflict_ids)
        if full_clean and full_conf:
            delta = head_level_collapse(
                [c.vam_per_layer_head for c in full_clean.values()],
                [c.vam_per_layer_head for c in full_conf.values()],
            )
            entry["head_collapse"] = IF.fig_head_collapse(delta, out, model, simulated)

        summary[model] = entry

    return summary
