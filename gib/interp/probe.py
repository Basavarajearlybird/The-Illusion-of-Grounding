"""Layer 3 -- the probe set and the headline experiment.

**The probe set.** ~500 conflict items and their 500 counterfactual twins, plus
matched clean controls. It is selected *after* the main eval, because the
selection depends on Layer 2's taxonomy labels: the headline comparison needs
items the model actually failed via Knowledge Inertia, and those are not known
until the model has been run.

**Matching.** Clean controls are matched to conflict items by ``pair_id``, not
sampled independently. VAM varies with prompt length and image content, so an
unmatched comparison would confound "attention collapsed under conflict" with
"these happened to be different images". Matching by pair_id makes the comparison
within-item, which is also what licenses the paired statistical test.

**The claim.** "During Knowledge Inertia, visual attention collapses" is
evaluated as a paired Wilcoxon signed-rank test with an effect size, not as a
pretty picture. The counterfactual twins add a second, harder test: does VAM
collapse *symmetrically* when the conflict flips, or is the text bias
object-dependent?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gib.config import INTERP_DIR, ensure_dirs
from gib.schema import Axis, FailureMode
from gib.seeds import rng_for


@dataclass
class ProbeSet:
    conflict_ids: list[str] = field(default_factory=list)
    twin_ids: list[str] = field(default_factory=list)
    clean_ids: list[str] = field(default_factory=list)
    inertia_ids: list[str] = field(default_factory=list)   # subset of conflict_ids

    def all_ids(self) -> list[str]:
        return sorted(set(self.conflict_ids) | set(self.twin_ids) | set(self.clean_ids))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> ProbeSet:
        return ProbeSet(**json.loads(Path(path).read_text(encoding="utf-8")))


def select_probe_set(
    results_rows: list[dict],
    variants: list[dict],
    *,
    n_conflict: int = 500,
    seed: int = 20260721,
) -> ProbeSet:
    """Choose the probe set from a model's results.

    Conflict items are sampled with Knowledge Inertia failures prioritised but
    correct items deliberately retained: the experiment is a *contrast*, and a
    probe set containing only failures would have nothing to contrast against.
    """
    by_variant = {v["variant_id"]: v for v in variants}

    conflict_rows = [
        r for r in results_rows if r.get("axis") == Axis.CONFLICT.value
    ]
    inertia = [
        r["variant_id"] for r in conflict_rows
        if r.get("failure_mode") == FailureMode.KNOWLEDGE_INERTIA.value
    ]
    other = [r["variant_id"] for r in conflict_rows if r["variant_id"] not in set(inertia)]

    rng = rng_for("probe_set", base=seed)
    n_inertia = min(len(inertia), int(n_conflict * 0.6))
    n_other = min(len(other), n_conflict - n_inertia)
    chosen = sorted(rng.sample(inertia, n_inertia) + rng.sample(other, n_other))

    # -- twins ------------------------------------------------------------- #
    twin_by_id: dict[str, list[str]] = {}
    for v in variants:
        if v.get("axis") == Axis.CONFLICT.value and v.get("twin_id"):
            twin_by_id.setdefault(v["twin_id"], []).append(v["variant_id"])

    twins: list[str] = []
    for vid in chosen:
        tid = by_variant.get(vid, {}).get("twin_id", "")
        for partner in twin_by_id.get(tid, []):
            if partner != vid:
                twins.append(partner)

    # -- matched clean controls, by pair_id -------------------------------- #
    clean_by_pair = {
        v["pair_id"]: v["variant_id"]
        for v in variants
        if v.get("axis") == Axis.CLEAN.value
    }
    clean = [
        clean_by_pair[by_variant[vid]["pair_id"]]
        for vid in chosen
        if vid in by_variant and by_variant[vid]["pair_id"] in clean_by_pair
    ]

    return ProbeSet(
        conflict_ids=chosen,
        twin_ids=sorted(set(twins)),
        clean_ids=sorted(set(clean)),
        inertia_ids=sorted(set(chosen) & set(inertia)),
    )


def run_probe(
    model: str,
    probe: ProbeSet,
    variants: list[dict],
    *,
    adapter_kwargs: dict | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the probe set with attention capture on and store per-item VAM.

    Per-item VAM tensors are stored as compressed npz keyed by ``variant_id``, so
    every figure in this layer regenerates from disk without re-running
    inference -- which matters because this is the slowest code path in the
    project.
    """
    from gib.adapters.registry import get_adapter, is_simulated
    from gib.interp.hooks import assert_eager_attention, vam_from_generation

    ensure_dirs()
    out_dir = Path(out_dir or INTERP_DIR) / model
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(model, **(adapter_kwargs or {}))
    if not adapter.capabilities.get("attention"):
        raise RuntimeError(
            f"{model} does not expose attention; Layer 3 runs on Track A models only."
        )
    if not is_simulated(model):
        assert_eager_attention(adapter)

    by_id = {v["variant_id"]: v for v in variants}
    wanted = probe.all_ids()
    written = 0
    missing = 0

    for i, vid in enumerate(wanted):
        row = by_id.get(vid)
        if row is None:
            missing += 1
            continue
        npz_path = out_dir / f"{vid}.npz"
        if npz_path.exists():
            continue  # resume-safe, same as Layer 2

        res = adapter.generate(
            row["image_path"], row["question"], list(row["retrieved_docs"] or []),
            capture_attention=True,
            meta={"variant_id": vid, "axis": row["axis"],
                  "answer_visual": row["answer_visual"],
                  "answer_textual": row["answer_textual"]},
        )
        cap = vam_from_generation(res.attn_capture or {}, vid, model)
        if cap is None:
            missing += 1
            continue
        cap.to_npz(npz_path)
        written += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"  [interp] {i+1}/{len(wanted)} captured")

    probe.save(out_dir / "probe_set.json")
    manifest = {
        "model": model,
        "n_requested": len(wanted),
        "n_written": written,
        "n_missing": missing,
        "n_conflict": len(probe.conflict_ids),
        "n_inertia": len(probe.inertia_ids),
        "n_twins": len(probe.twin_ids),
        "n_clean": len(probe.clean_ids),
    }
    (out_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    adapter.close()
    if verbose:
        print(f"[interp] {model}: {written} captures -> {out_dir}")
    return manifest


def load_captures(model: str, ids: list[str], out_dir: Path | None = None) -> dict[str, np.ndarray]:
    """variant_id -> per-layer VAM vector, for the ids that were captured."""
    from gib.interp.hooks import VAMCapture

    base = Path(out_dir or INTERP_DIR) / model
    out: dict[str, np.ndarray] = {}
    for vid in ids:
        p = base / f"{vid}.npz"
        if p.exists():
            out[vid] = VAMCapture.from_npz(p, vid, model).vam_per_layer
    return out


def load_captures_full(model: str, ids: list[str], out_dir: Path | None = None):
    from gib.interp.hooks import VAMCapture

    base = Path(out_dir or INTERP_DIR) / model
    return {
        vid: VAMCapture.from_npz(base / f"{vid}.npz", vid, model)
        for vid in ids
        if (base / f"{vid}.npz").exists()
    }
