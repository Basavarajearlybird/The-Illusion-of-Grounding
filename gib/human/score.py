"""Scoring the human study, and the model-vs-human gap.

Reported per the architecture: per-stressor accuracy, inter-rater agreement
(Krippendorff's alpha), and median response time.

The paper's target framing -- humans ~90-95% where GPT-4o falls to 40-60% -- is
plausible specifically on the *conflict* axis, since humans trivially trust their
eyes over a contradicting paragraph. The stratified design is what lets the paper
show where the gap opens and where, at extreme blur, it honestly narrows.
Reporting the narrowing case is what makes the study credible, so
:func:`gap_by_stressor` returns every cell rather than only the flattering ones.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gib.config import HUMAN_DIR
from gib.eval.scoring import score_cascade
from gib.eval.taxonomy import TaxonomyInput, classify


@dataclass
class HumanResponse:
    participant_id: str
    variant_id: str
    answer: str
    response_time_s: float
    timestamp: str = ""


@dataclass
class ScoredResponse:
    participant_id: str
    variant_id: str
    axis: str
    dose: str
    sub_type: str
    answer: str
    is_correct: bool
    matches_textual: bool
    failure_mode: str
    response_time_s: float


def score_responses(
    responses: list[HumanResponse], items_by_id: dict, judge=None
) -> list[ScoredResponse]:
    """Score with the identical cascade used for models."""
    out: list[ScoredResponse] = []
    for r in responses:
        item = items_by_id.get(r.variant_id)
        if item is None:
            continue
        av = item["answer_visual"] if isinstance(item, dict) else item.answer_visual
        at = item["answer_textual"] if isinstance(item, dict) else item.answer_textual
        q = item["question"] if isinstance(item, dict) else item.question
        axis = item["axis"] if isinstance(item, dict) else item.axis
        dose = item["dose"] if isinstance(item, dict) else item.dose
        st = item["sub_type"] if isinstance(item, dict) else item.sub_type

        sv = score_cascade(r.answer, av, judge, question=q)
        stx = score_cascade(r.answer, at, judge, question=q) if at else None
        fm = classify(
            TaxonomyInput(
                axis=axis, matches_visual=sv.matched,
                matches_textual=bool(stx and stx.matched), prediction=r.answer,
                answer_textual=at,
                degradation="gaussian_blur" if axis in ("visual", "combined") else "none",
            )
        )
        out.append(
            ScoredResponse(
                participant_id=r.participant_id, variant_id=r.variant_id, axis=axis,
                dose=dose, sub_type=st, answer=r.answer, is_correct=sv.matched,
                matches_textual=bool(stx and stx.matched), failure_mode=fm.value,
                response_time_s=r.response_time_s,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def krippendorff_alpha(
    ratings: dict[str, dict[str, str]], level: str = "nominal"
) -> float:
    """Krippendorff's alpha over ``{item: {rater: label}}``.

    Krippendorff rather than Fleiss' kappa because this design has *missing
    data* by construction -- not every participant sees every item -- and Fleiss
    requires a complete rating matrix.
    """
    items = [r for r in ratings.values() if len(r) >= 2]
    if not items:
        return float("nan")

    values = sorted({v for r in items for v in r.values()})
    if len(values) < 2:
        return 1.0
    vidx = {v: i for i, v in enumerate(values)}
    k = len(values)

    coincidence = np.zeros((k, k), dtype=float)
    for r in items:
        labels = list(r.values())
        m = len(labels)
        if m < 2:
            continue
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i != j:
                    coincidence[vidx[a], vidx[b]] += 1.0 / (m - 1)

    n = coincidence.sum()
    if n == 0:
        return float("nan")
    nc = coincidence.sum(axis=1)

    if level == "nominal":
        delta = 1.0 - np.eye(k)
    else:  # interval, over the ordinal index
        idx = np.arange(k, dtype=float)
        delta = (idx[:, None] - idx[None, :]) ** 2

    do = (coincidence * delta).sum() / n
    de = (np.outer(nc, nc) * delta).sum() / (n * (n - 1)) if n > 1 else 0.0
    return float(1.0 - do / de) if de > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass
class HumanSummary:
    n_participants: int
    n_responses: int
    overall_accuracy: float
    by_axis: dict = field(default_factory=dict)
    by_axis_dose: dict = field(default_factory=dict)
    by_sub_type: dict = field(default_factory=dict)
    median_rt_s: float = float("nan")
    krippendorff_alpha: float = float("nan")
    taxonomy: dict = field(default_factory=dict)


def summarize(scored: list[ScoredResponse]) -> HumanSummary:
    if not scored:
        return HumanSummary(0, 0, float("nan"))

    def acc(rows):
        return float(np.mean([r.is_correct for r in rows])) if rows else float("nan")

    by_axis: dict = defaultdict(list)
    by_axis_dose: dict = defaultdict(list)
    by_sub: dict = defaultdict(list)
    for r in scored:
        by_axis[r.axis].append(r)
        by_axis_dose[f"{r.axis}|{r.dose}"].append(r)
        by_sub[r.sub_type].append(r)

    ratings: dict[str, dict[str, str]] = defaultdict(dict)
    for r in scored:
        ratings[r.variant_id][r.participant_id] = r.answer.strip().lower()

    tax: dict[str, int] = defaultdict(int)
    for r in scored:
        tax[r.failure_mode] += 1

    return HumanSummary(
        n_participants=len({r.participant_id for r in scored}),
        n_responses=len(scored),
        overall_accuracy=acc(scored),
        by_axis={k: {"accuracy": acc(v), "n": len(v)} for k, v in sorted(by_axis.items())},
        by_axis_dose={k: {"accuracy": acc(v), "n": len(v)} for k, v in sorted(by_axis_dose.items())},
        by_sub_type={k: {"accuracy": acc(v), "n": len(v)} for k, v in sorted(by_sub.items())},
        median_rt_s=float(np.median([r.response_time_s for r in scored])),
        krippendorff_alpha=krippendorff_alpha(dict(ratings)),
        taxonomy={k: v / len(scored) for k, v in sorted(tax.items())},
    )


def gap_by_stressor(human: HumanSummary, model_acc_by_axis_dose: dict[str, float]) -> list[dict]:
    """Human-minus-model accuracy for every stressor cell both cover.

    Sorted by gap descending, so the conflict cells that carry the paper's
    headline surface first -- but the narrowing cells are still in the table.
    """
    rows = []
    for key, stats in human.by_axis_dose.items():
        if key not in model_acc_by_axis_dose:
            continue
        h = stats["accuracy"]
        m = model_acc_by_axis_dose[key]
        axis, dose = key.split("|", 1)
        rows.append(
            {
                "axis": axis, "dose": dose, "human_accuracy": round(h, 4),
                "model_accuracy": round(m, 4), "gap": round(h - m, 4), "n_human": stats["n"],
            }
        )
    return sorted(rows, key=lambda r: -r["gap"])


def save_summary(summary: HumanSummary, path: Path | None = None) -> Path:
    path = Path(path or HUMAN_DIR / "human_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_participants": summary.n_participants,
        "n_responses": summary.n_responses,
        "overall_accuracy": summary.overall_accuracy,
        "by_axis": summary.by_axis,
        "by_axis_dose": summary.by_axis_dose,
        "by_sub_type": summary.by_sub_type,
        "median_rt_s": summary.median_rt_s,
        "krippendorff_alpha": summary.krippendorff_alpha,
        "taxonomy": summary.taxonomy,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_responses(path: Path | None = None, *, allow_simulated: bool = False) -> list[HumanResponse]:
    """Load participant responses from the JSONL the Streamlit app writes.

    Simulated responses live in a separate file and are only loaded when
    explicitly requested, so a real scoring run can never silently pick them up.
    """
    if path is None:
        real = HUMAN_DIR / "responses.jsonl"
        if real.exists():
            path = real
        elif allow_simulated:
            path = HUMAN_DIR / "responses_simulated.jsonl"
            if path.exists():
                print("[human] *** using SIMULATED responses -- not a human baseline ***")
        else:
            path = real
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(
                HumanResponse(
                    participant_id=d["participant_id"], variant_id=d["variant_id"],
                    answer=d["answer"], response_time_s=float(d["response_time_s"]),
                    timestamp=d.get("timestamp", ""),
                )
            )
    return out
