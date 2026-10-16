"""Layer 1 orchestrator: sources -> clean pairs -> variants -> parquet.

Produces two splits, per the architecture's instruction not to evaluate the full
cross-product:

* ``GIB-15K-eval``  -- the canonical split the paper reports (~45-60K rows).
* ``GIB-15K-full``  -- the superset, released for downstream users.

The eval split is *stratified* by axis x dose x sub_type rather than sampled
uniformly, because the vulnerability heatmap and the per-cell taxonomy
proportions both need a floor on n per cell; a uniform sample would starve the
rare cells while over-filling the common ones.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from gib.config import (
    PAIRS_PATH,
    VARIANTS_DIR,
    DatasetConfig,
    ensure_dirs,
    git_sha,
)
from gib.dataengine.conflict import make_clean_variant, make_conflict_variant, plan_conflicts
from gib.dataengine.degrade import make_combined_variants, make_visual_variants
from gib.dataengine.hardneg import build_pools, make_noise_variants
from gib.dataengine.qagen import generate_pairs, spot_check_sample
from gib.dataengine.sources import get_source
from gib.schema import Axis, CleanPair, SubType, Variant, dataset_hash
from gib.seeds import rng_for


def _environment() -> dict:
    """Capture the environment for the manifest — enough to reconstruct the
    build conditions when a number is later questioned."""
    import platform
    import sys

    versions = {}
    for mod in ("numpy", "pandas", "pyarrow", "cv2", "sklearn", "scipy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _iter_scenes(cfg: DatasetConfig, scene_limit: int):
    """Interleave the configured sources.

    Interleaving rather than concatenating matters when a build is truncated:
    a concatenated stream truncated early would yield an all-COCO corpus and
    silently lose the attribute and relation sub-types that only VG supplies.
    """
    iters = []
    for name in cfg.sources:
        try:
            iters.append(iter(get_source(name)(limit=scene_limit, seed_base=cfg.seed)))
        except FileNotFoundError as exc:
            print(f"  [build] source {name!r} unavailable: {exc}")
    if not iters:
        raise RuntimeError(
            f"no usable sources in {cfg.sources}. Use sources: [synthetic] for an offline build."
        )
    alive = True
    while alive:
        alive = False
        for it in iters:
            nxt = next(it, None)
            if nxt is not None:
                alive = True
                yield nxt


def build_clean_pairs(cfg: DatasetConfig, verbose: bool = True) -> tuple[list[CleanPair], dict]:
    """Stage 1: ingest sources, template QA, round-trip validate."""
    t0 = time.time()
    # Generous scene budget: round-trip validation and uniqueness filtering both
    # reject, so scenes-in must exceed pairs-out by a wide margin.
    scene_limit = max(cfg.n_pairs, cfg.per_sub_type * 3)
    scenes = _iter_scenes(cfg, scene_limit)
    pairs, stats = generate_pairs(
        scenes, per_sub_type=cfg.per_sub_type, seed_base=cfg.seed, paraphrase_rate=0.0
    )
    stats["elapsed_s"] = round(time.time() - t0, 1)
    if verbose:
        by_st = defaultdict(int)
        for p in pairs:
            by_st[p.sub_type.value] += 1
        print(f"  [build] clean pairs: {len(pairs)} {dict(by_st)}  ({stats['elapsed_s']}s)")
        print(
            f"  [build] round-trip rejected {stats['round_trip_rejected']} of "
            f"{stats['candidates']} candidates"
        )
    return pairs, stats


def build_variants(
    pairs: list[CleanPair], cfg: DatasetConfig, verbose: bool = True
) -> tuple[list[Variant], dict]:
    """Stage 2: run all three adversarial generators plus the combined sub-grid."""
    info: dict = {}
    variants: list[Variant] = []

    # -- clean control ------------------------------------------------------- #
    variants.extend(make_clean_variant(p, cfg.seed) for p in pairs)

    # -- Axis 1 -------------------------------------------------------------- #
    t0 = time.time()
    pools, embed_info = build_pools(
        pairs, embedder_name=cfg.noise.embedder, top_k=cfg.noise.top_k_neighbours,
        seed_base=cfg.seed, verbose=verbose,
    )
    info.update(embed_info)
    for p in pairs:
        variants.extend(
            make_noise_variants(
                p, pools[p.pair_id], doses=cfg.noise.doses,
                hard_ratio=cfg.noise.hard_negative_ratio, seed_base=cfg.seed,
            )
        )
    if verbose:
        print(f"  [build] axis 1 (noise) done in {time.time()-t0:.1f}s")

    # -- Axis 2 -------------------------------------------------------------- #
    t0 = time.time()
    for i, p in enumerate(pairs):
        variants.extend(
            make_visual_variants(
                p,
                gaussian_sigma=cfg.visual.gaussian_sigma,
                motion_kernel=cfg.visual.motion_kernel,
                crop_visible=cfg.visual.crop_visible,
                native_resolution=cfg.visual.native_resolution,
                seed_base=cfg.seed,
            )
        )
        if verbose and (i + 1) % 500 == 0:
            print(f"  [build] axis 2 (visual): {i+1}/{len(pairs)} pairs degraded")
    if verbose:
        print(f"  [build] axis 2 (visual) done in {time.time()-t0:.1f}s")

    # -- Axis 3 -------------------------------------------------------------- #
    t0 = time.time()
    specs = plan_conflicts(pairs, make_twins=cfg.conflict.make_twins, seed_base=cfg.seed)
    conflict_variants = [make_conflict_variant(s, cfg.seed) for s in specs]
    variants.extend(conflict_variants)
    n_twinned = sum(1 for v in conflict_variants if v.twin_id)
    info["conflict_items"] = len(conflict_variants)
    info["conflict_twinned"] = n_twinned
    if verbose:
        print(
            f"  [build] axis 3 (conflict): {len(conflict_variants)} items, "
            f"{n_twinned} twinned ({n_twinned/max(1,len(conflict_variants)):.0%}) "
            f"in {time.time()-t0:.1f}s"
        )

    # -- combined sub-grid (heatmap only) ------------------------------------ #
    rng = rng_for("combined_subset", base=cfg.seed)
    subset = rng.sample(pairs, min(cfg.combined.n_pairs, len(pairs)))
    for p in subset:
        variants.extend(
            make_combined_variants(
                p, pools[p.pair_id],
                sigma_grid=cfg.combined.sigma_grid,
                distractor_grid=cfg.combined.distractor_grid,
                hard_ratio=cfg.noise.hard_negative_ratio,
                seed_base=cfg.seed,
            )
        )
    info["combined_pairs"] = len(subset)
    if verbose:
        print(f"  [build] combined sub-grid: {len(subset)} pairs")

    return variants, info


def stratified_eval_split(
    variants: list[Variant], target: int, seed: int, min_per_cell: int = 50
) -> list[Variant]:
    """Sample the canonical eval split, stratified by axis x dose x sub_type.

    Cells smaller than ``min_per_cell`` are taken whole -- shrinking an already
    thin cell to hit a global row target would be exactly the wrong trade.
    """
    cells: dict[tuple, list[Variant]] = defaultdict(list)
    for v in variants:
        cells[(v.axis.value, v.dose, v.sub_type.value)].append(v)

    # Clean rows are the denominator for every ratio in Layer 5, so they are
    # never subsampled.
    keep: list[Variant] = []
    sampled_cells = {}
    for key, rows in cells.items():
        if key[0] == Axis.CLEAN.value:
            keep.extend(rows)
        else:
            sampled_cells[key] = rows

    budget = max(0, target - len(keep))
    n_cells = max(1, len(sampled_cells))
    per_cell = max(min_per_cell, budget // n_cells)

    for key in sorted(sampled_cells):
        rows = sorted(sampled_cells[key], key=lambda v: v.variant_id)
        rng = rng_for("eval_split", *key, base=seed)
        keep.extend(rows if len(rows) <= per_cell else rng.sample(rows, per_cell))

    # Counterfactual twins must survive sampling *as pairs*. Sampling rows
    # independently orphans most twins (a pair survives only if both halves are
    # drawn), which would silently make the counterfactual-symmetry analysis
    # uncomputable on the canonical split -- the twin_id column would still be
    # populated, so the loss would not be obvious.
    kept_ids = {v.variant_id for v in keep}
    by_twin: dict[str, list[Variant]] = defaultdict(list)
    for v in variants:
        if v.twin_id:
            by_twin[v.twin_id].append(v)
    for group in by_twin.values():
        if any(v.variant_id in kept_ids for v in group):
            for v in group:
                if v.variant_id not in kept_ids:
                    keep.append(v)
                    kept_ids.add(v.variant_id)

    return sorted(keep, key=lambda v: v.variant_id)


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def build(
    config_path: str = "dataset.yaml",
    *,
    n_pairs: int | None = None,
    sources: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Full Layer 1 build. Returns the manifest."""
    ensure_dirs()
    cfg = DatasetConfig.from_yaml(config_path)
    if n_pairs is not None:
        cfg.n_pairs = n_pairs
        cfg.per_sub_type = max(1, n_pairs // 3)
        cfg.combined.n_pairs = min(cfg.combined.n_pairs, max(1, n_pairs // 15))
        cfg.eval_split_size = max(1000, n_pairs * 4)
    if sources is not None:
        cfg.sources = sources

    print(f"[GIB-15K] building: {cfg.n_pairs} pairs from {cfg.sources}")
    pairs, qa_stats = build_clean_pairs(cfg, verbose=verbose)
    if not pairs:
        raise RuntimeError("no clean pairs survived generation; check source availability")

    write_parquet([p.to_row() for p in pairs], PAIRS_PATH)

    variants, var_info = build_variants(pairs, cfg, verbose=verbose)
    rows = [v.to_row() for v in variants]

    # -- write one parquet per axis (the eval engine partitions by axis) ----- #
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r)
    for axis, axis_rows in by_axis.items():
        write_parquet(axis_rows, VARIANTS_DIR / f"axis={axis}" / "data.parquet")

    write_parquet(rows, VARIANTS_DIR / "gib15k_full.parquet")

    eval_rows = stratified_eval_split(variants, cfg.eval_split_size, cfg.seed)
    write_parquet([v.to_row() for v in eval_rows], VARIANTS_DIR / "gib15k_eval.parquet")

    spot = spot_check_sample(pairs, rate=0.05, seed_base=cfg.seed)
    write_parquet([p.to_row() for p in spot], VARIANTS_DIR / "spot_check_sample.parquet")

    manifest = {
        "schema_version": "1.0",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "seed": cfg.seed,
        "sources": cfg.sources,
        "environment": _environment(),
        "n_clean_pairs": len(pairs),
        "n_variants_full": len(rows),
        "n_variants_eval": len(eval_rows),
        "dataset_hash": dataset_hash(rows),
        "eval_hash": dataset_hash([v.to_row() for v in eval_rows]),
        "by_axis": {k: len(v) for k, v in sorted(by_axis.items())},
        "by_sub_type": {
            st.value: sum(1 for p in pairs if p.sub_type is st) for st in SubType
        },
        "spot_check_n": len(spot),
        "qa_stats": qa_stats,
        **var_info,
    }
    (VARIANTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # -- measured characterisation (feeds the datasheet + paper_numbers) ------ #
    try:
        from gib.dataengine.characterize import characterize, to_markdown

        report = characterize([v.to_row() for v in eval_rows], [p.to_row() for p in pairs])
        (VARIANTS_DIR / "characterization.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (VARIANTS_DIR / "characterization.md").write_text(to_markdown(report), encoding="utf-8")
        manifest["characterization"] = {
            k: report[k] for k in (
                "sub_type_balance_entropy", "answer_entropy",
                "hard_negative_overlap", "conflict", "twin_yield",
            ) if k in report
        }
        (VARIANTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if verbose and "hard_negative_overlap" in report:
            print(f"  [build] hard-negative difficulty (Jaccard): "
                  f"mean {report['hard_negative_overlap']['mean']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [build] characterisation skipped: {type(exc).__name__}: {exc}")

    print(f"[GIB-15K] {len(pairs)} pairs -> {len(rows)} variants "
          f"(eval split: {len(eval_rows)})")
    print(f"[GIB-15K] manifest: {VARIANTS_DIR / 'manifest.json'}")
    return manifest
