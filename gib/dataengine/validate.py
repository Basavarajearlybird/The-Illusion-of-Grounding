"""Dataset contract validation.

The five layers are connected by versioned data contracts, and this is the check
that the contract actually holds. It is cheap and it catches the failures that
would otherwise be discovered only after a multi-day evaluation campaign had
already run against a broken build:

* an ``answer_textual`` equal to ``answer_visual`` -- a "conflict" item with no
  conflict in it, which would be scored as both correct and Knowledge Inertia;
* a conflict item whose retrieved passage does not actually contain the
  falsehood;
* a noise item missing its ground-truth document, which by construction must
  always be present;
* image hashes that no longer match the files on disk;
* twin pairs that are not genuine mutual flips.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from gib.config import VARIANTS_DIR
from gib.schema import Axis, sha256_file


def validate_dataset(dataset: str = "gib15k_eval.parquet", *, check_hashes: int = 200) -> dict:
    """Returns a report; ``ok`` is False if any hard invariant is violated."""
    import pyarrow.parquet as pq

    path = VARIANTS_DIR / dataset
    if not path.exists():
        return {"ok": False, "errors": [f"dataset not found: {path}"]}

    rows = pq.read_table(path).to_pylist()
    errors: list[str] = []
    warnings: list[str] = []

    # -- uniqueness ---------------------------------------------------------- #
    ids = Counter(r["variant_id"] for r in rows)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        errors.append(f"{len(dupes)} duplicate variant_ids (e.g. {dupes[:3]})")

    # -- per-axis invariants -------------------------------------------------- #
    n_no_conflict = 0
    n_falsehood_missing = 0
    n_gt_missing = 0
    n_empty_q = 0

    for r in rows:
        axis = r["axis"]
        if not r["question"].strip() or not r["answer_visual"].strip():
            n_empty_q += 1

        if axis == Axis.CONFLICT.value:
            if not r["answer_textual"]:
                n_no_conflict += 1
            elif r["answer_textual"].strip().lower() == r["answer_visual"].strip().lower():
                n_no_conflict += 1
            else:
                docs = " ".join(r["retrieved_docs"] or []).lower()
                if r["answer_textual"].strip().lower() not in docs:
                    n_falsehood_missing += 1

        if axis == Axis.NOISE.value:
            if r["gt_doc_position"] < 0:
                n_gt_missing += 1
            n_docs = len(r["retrieved_docs"] or [])
            if n_docs != int(r["n_distractors"]) + 1:
                warnings.append(
                    f"{r['variant_id']}: {n_docs} docs for dose {r['n_distractors']}"
                )

    if n_no_conflict:
        errors.append(f"{n_no_conflict} conflict rows where textual == visual (no conflict)")
    if n_falsehood_missing:
        errors.append(f"{n_falsehood_missing} conflict rows whose passage omits the falsehood")
    if n_gt_missing:
        errors.append(f"{n_gt_missing} noise rows missing the ground-truth document")
    if n_empty_q:
        errors.append(f"{n_empty_q} rows with an empty question or visual answer")

    # -- twins ---------------------------------------------------------------- #
    twins: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("twin_id"):
            twins[r["twin_id"]].append(r)
    # Two distinct problems, kept distinct: an *orphaned* twin means the split
    # dropped a partner (symmetry becomes uncomputable), whereas a *broken* twin
    # means the generator paired two items that are not mutual flips. Collapsing
    # them into one warning hides which one is happening.
    orphaned = [tid for tid, g in twins.items() if len(g) != 2]
    broken = []
    for tid, group in twins.items():
        if len(group) != 2:
            continue
        a, b = group
        # A genuine counterfactual: each one's falsehood is the other's truth.
        if not (
            a["answer_textual"].lower() == b["answer_visual"].lower()
            and b["answer_textual"].lower() == a["answer_visual"].lower()
        ):
            broken.append(tid)
    if broken:
        errors.append(f"{len(broken)} twin groups are not exact mutual flips")
    if orphaned:
        warnings.append(
            f"{len(orphaned)} twin groups have a missing partner in this split; "
            f"counterfactual symmetry cannot be computed for them"
        )

    # -- image hashes (sampled) ------------------------------------------------ #
    import random

    rng = random.Random(20260721)
    sample = rng.sample(rows, min(check_hashes, len(rows)))
    n_missing_img = 0
    n_bad_hash = 0
    for r in sample:
        p = Path(r["image_path"])
        if not p.exists():
            n_missing_img += 1
        elif sha256_file(str(p)) != r["image_sha256"]:
            n_bad_hash += 1
    if n_missing_img:
        errors.append(f"{n_missing_img}/{len(sample)} sampled images missing on disk")
    if n_bad_hash:
        errors.append(f"{n_bad_hash}/{len(sample)} sampled images do not match their sha256")

    by_axis = Counter(r["axis"] for r in rows)
    by_sub = Counter(r["sub_type"] for r in rows)

    return {
        "ok": not errors,
        "dataset": str(path),
        "n_rows": len(rows),
        "by_axis": dict(sorted(by_axis.items())),
        "by_sub_type": dict(sorted(by_sub.items())),
        "n_twin_groups": len(twins),
        "images_checked": len(sample),
        "errors": errors,
        "warnings": warnings[:20],
        "n_warnings": len(warnings),
    }
