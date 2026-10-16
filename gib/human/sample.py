"""Layer 4 -- human study sampling and scoring.

Deliberately small and clean: ~300 items stratified by axis x dose x sub_type,
drawn from the **exact adversarial split the models saw** -- same degraded
images, same noisy passages, byte-identical. The byte-identity is enforced rather
than assumed (:func:`verify_identical`), by re-hashing the served images against
``image_sha256`` from the dataset. If the humans and the models see even slightly
different stimuli, the headline gap is not a finding.

Humans are scored with the *same* judge cascade as the models. That symmetry is
what makes the comparison legitimate; scoring humans leniently by hand and models
strictly by code would manufacture the gap the paper is trying to measure.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gib.config import HUMAN_DIR, VARIANTS_DIR
from gib.schema import sha256_file
from gib.seeds import rng_for


@dataclass
class HumanItem:
    """One item as served to a participant. Answers are never included."""

    variant_id: str
    pair_id: str
    axis: str
    dose: str
    sub_type: str
    image_path: str
    image_sha256: str
    question: str
    retrieved_docs: list[str] = field(default_factory=list)
    # kept server-side for scoring, stripped before serving
    answer_visual: str = ""
    answer_textual: str = ""


def build_study_sample(
    dataset: str = "gib15k_eval.parquet",
    *,
    n_items: int = 300,
    seed: int = 20260721,
    include_clean: bool = True,
) -> list[HumanItem]:
    """Stratified sample across axis x dose x sub_type.

    The stratification is the whole design. It is what lets the paper show
    *where* the human-model gap opens (conflict) and where it honestly narrows
    (extreme blur) -- a uniform sample would report one average number and could
    show neither.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(VARIANTS_DIR / dataset).to_pylist()
    if not include_clean:
        rows = [r for r in rows if r["axis"] != "clean"]

    # Allocate the budget across AXES first, then within each axis across its
    # dose x sub_type cells.
    #
    # Allocating equally per cell instead would silently hand the study to
    # whichever axis has the most dose levels: the combined sub-grid has 75 cells
    # (5 sigma x 5 distractor counts x 3 sub-types) against conflict's 3, so a
    # flat per-cell budget gives ~57% of a 300-item study to combined and leaves
    # conflict with ~25 items. Conflict is where the paper's headline gap lives,
    # and n=25 cannot support it.
    by_axis: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_axis[r["axis"]][(r["dose"], r["sub_type"])].append(r)

    axes = sorted(by_axis)
    per_axis = max(1, n_items // max(1, len(axes)))

    out: list[HumanItem] = []
    for axis in axes:
        cells = by_axis[axis]
        per_cell = max(1, per_axis // max(1, len(cells)))
        taken: list[dict] = []
        for key in sorted(cells):
            group = sorted(cells[key], key=lambda r: r["variant_id"])
            rng = rng_for("human_sample", axis, *key, base=seed)
            taken.extend(group if len(group) <= per_cell else rng.sample(group, per_cell))

        # Top up or trim so each axis lands near its share even when its cells
        # are too few or too many to hit it exactly.
        rng = rng_for("human_axis_balance", axis, base=seed)
        if len(taken) > per_axis:
            taken = rng.sample(taken, per_axis)
        elif len(taken) < per_axis:
            pool = [r for c in cells.values() for r in c
                    if r["variant_id"] not in {t["variant_id"] for t in taken}]
            if pool:
                taken.extend(rng.sample(pool, min(per_axis - len(taken), len(pool))))

        for r in taken:
            out.append(
                HumanItem(
                    variant_id=r["variant_id"], pair_id=r["pair_id"], axis=r["axis"],
                    dose=r["dose"], sub_type=r["sub_type"], image_path=r["image_path"],
                    image_sha256=r["image_sha256"], question=r["question"],
                    retrieved_docs=list(r["retrieved_docs"] or []),
                    answer_visual=r["answer_visual"], answer_textual=r["answer_textual"],
                )
            )

    return sorted(out, key=lambda i: i.variant_id)


def verify_identical(items: list[HumanItem]) -> list[str]:
    """Re-hash every served image against the dataset. Returns mismatched ids.

    Cheap insurance against the single most damaging silent bug in this layer:
    serving a re-encoded or resized image to humans while the models saw the
    original.
    """
    bad = []
    for it in items:
        p = Path(it.image_path)
        if not p.exists() or sha256_file(str(p)) != it.image_sha256:
            bad.append(it.variant_id)
    return bad


def save_study(items: list[HumanItem], path: Path | None = None) -> Path:
    path = Path(path or HUMAN_DIR / "study_items.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(i) for i in items], indent=2), encoding="utf-8")
    return path


def load_study(path: Path | None = None) -> list[HumanItem]:
    path = Path(path or HUMAN_DIR / "study_items.json")
    return [HumanItem(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
