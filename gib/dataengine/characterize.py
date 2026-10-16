"""Dataset characterisation — the measured quality report for the datasheet.

A datasheet that only *asserts* balance and difficulty is a template; one that
*measures* them is evidence. This module computes the statistics a benchmark
reviewer actually asks for, straight off the built parquet, and emits them as
both JSON (for `paper_numbers.json`) and a Markdown section (for the datasheet).

Every number here is a property a reviewer can attack if it is wrong, so each is
computed from the data rather than restated from the config:

* **Sub-type balance** — are object/attribute/relation actually even?
* **Hard-negative difficulty** — how lexically close are the injected distractors
  to the ground-truth passage? A benchmark whose "hard" negatives share few words
  with the truth is not hard. Measured as Jaccard token overlap; higher = harder.
* **Conflict plausibility** — do entity swaps stay within the WikiData sibling
  class? An out-of-class swap ("apple"->"helicopter") is an easy item.
* **Twin yield** — the fraction of conflict items with an exact counterfactual
  partner, which bounds the counterfactual-symmetry analysis.
* **Answer entropy** — is the answer distribution diverse, or does one label
  dominate (which a model could exploit as a prior)?
* **Degradation coverage** — every dose present at the expected count.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from gib.dataengine.sources.wikidata import ENTITY_TO_CLASS


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _entropy(counts: Counter) -> float:
    """Shannon entropy in bits; normalised to [0,1] by log2(#classes)."""
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c)
    return h / math.log2(len(counts))


def characterize(variants: list[dict], pairs: list[dict] | None = None) -> dict[str, Any]:
    """Compute the full characterisation report from variant rows."""
    report: dict[str, Any] = {}
    n = len(variants)
    report["n_variants"] = n

    # -- sub-type balance ---------------------------------------------------- #
    by_sub = Counter(v["sub_type"] for v in variants)
    report["sub_type_counts"] = dict(sorted(by_sub.items()))
    report["sub_type_balance_entropy"] = round(_entropy(by_sub), 4)

    # -- answer diversity ---------------------------------------------------- #
    answers = Counter(v["answer_visual"].lower() for v in variants if v["answer_visual"])
    report["n_distinct_visual_answers"] = len(answers)
    report["answer_entropy"] = round(_entropy(answers), 4)
    report["top_answers"] = dict(answers.most_common(10))
    top_share = answers.most_common(1)[0][1] / max(1, sum(answers.values()))
    report["most_common_answer_share"] = round(top_share, 4)

    # -- hard-negative difficulty (noise axis) ------------------------------- #
    # For every noise row, the lexical overlap between the ground-truth passage
    # and its hardest distractor. Higher overlap = a harder negative.
    overlaps = []
    for v in variants:
        if v["axis"] != "noise":
            continue
        docs = list(v.get("retrieved_docs") or [])
        pos = v.get("gt_doc_position", -1)
        if pos < 0 or pos >= len(docs) or len(docs) < 2:
            continue
        gt = docs[pos]
        distractors = [d for i, d in enumerate(docs) if i != pos]
        # The hardest distractor is the one most lexically similar to the truth.
        overlaps.append(max(_jaccard(gt, d) for d in distractors))
    if overlaps:
        overlaps.sort()
        report["hard_negative_overlap"] = {
            "mean": round(sum(overlaps) / len(overlaps), 4),
            "median": round(overlaps[len(overlaps) // 2], 4),
            "p90": round(overlaps[int(len(overlaps) * 0.9)], 4),
            "n": len(overlaps),
        }

    # -- conflict plausibility ----------------------------------------------- #
    conflict = [v for v in variants if v["axis"] == "conflict"]
    in_class = same = 0
    for v in conflict:
        vis, txt = v["answer_visual"].lower(), v["answer_textual"].lower()
        if not txt or txt == vis:
            same += 1
            continue
        cv, ct = ENTITY_TO_CLASS.get(vis), ENTITY_TO_CLASS.get(txt)
        # Plausible if the swap stayed in the sibling class, OR the swap was an
        # attribute/predicate flip (which has no entity class but is family-bound
        # by construction in the mutator).
        if (cv and cv == ct) or v["sub_type"] in ("attribute", "relation"):
            in_class += 1
    report["conflict"] = {
        "n": len(conflict),
        "plausible_swap_rate": round(in_class / max(1, len(conflict)), 4),
        "degenerate_no_conflict": same,
    }

    # -- twin yield ---------------------------------------------------------- #
    twins = Counter(v["twin_id"] for v in conflict if v.get("twin_id"))
    complete = sum(1 for c in twins.values() if c == 2)
    report["twin_yield"] = {
        "n_twinned_items": sum(1 for v in conflict if v.get("twin_id")),
        "n_complete_pairs": complete,
        "twinned_fraction": round(
            sum(1 for v in conflict if v.get("twin_id")) / max(1, len(conflict)), 4
        ),
    }

    # -- degradation coverage ------------------------------------------------ #
    visual = [v for v in variants if v["axis"] == "visual"]
    report["visual_dose_coverage"] = dict(sorted(Counter(v["dose"] for v in visual).items()))
    noise = [v for v in variants if v["axis"] == "noise"]
    report["noise_dose_coverage"] = dict(
        sorted(Counter(v["dose"] for v in noise).items(), key=lambda kv: int(kv[0]))
    )

    # -- source provenance --------------------------------------------------- #
    report["source_counts"] = dict(sorted(Counter(v.get("source", "") for v in variants).items()))

    return report


def to_markdown(report: dict) -> str:
    """Render the report as a datasheet section."""
    lines = ["## Measured dataset characterisation", ""]
    lines.append(f"- Variant rows: **{report['n_variants']:,}**")
    lines.append(
        f"- Sub-type balance (normalised entropy, 1.0 = perfectly even): "
        f"**{report['sub_type_balance_entropy']}** — {report['sub_type_counts']}"
    )
    lines.append(
        f"- Distinct visual answers: **{report['n_distinct_visual_answers']}**, "
        f"answer entropy **{report['answer_entropy']}**, "
        f"most-common-answer share **{report['most_common_answer_share']:.1%}**"
    )
    if "hard_negative_overlap" in report:
        h = report["hard_negative_overlap"]
        lines.append(
            f"- Hard-negative difficulty (token Jaccard between truth and hardest "
            f"distractor; higher = harder): mean **{h['mean']}**, median "
            f"**{h['median']}**, p90 **{h['p90']}** (n={h['n']:,})"
        )
    c = report["conflict"]
    lines.append(
        f"- Conflict plausibility: **{c['plausible_swap_rate']:.1%}** of swaps stay "
        f"in-class/family; {c['degenerate_no_conflict']} degenerate (no conflict)"
    )
    t = report["twin_yield"]
    lines.append(
        f"- Counterfactual twins: **{t['n_complete_pairs']}** complete pairs "
        f"(**{t['twinned_fraction']:.1%}** of conflict items twinned)"
    )
    lines.append(f"- Source provenance: {report['source_counts']}")
    lines.append("")
    return "\n".join(lines)


def run(dataset: str = "gib15k_eval.parquet", variants_dir: Path | None = None) -> dict:
    import pyarrow.parquet as pq

    from gib.config import VARIANTS_DIR

    base = Path(variants_dir or VARIANTS_DIR)
    variants = pq.read_table(base / dataset).to_pylist()
    report = characterize(variants)

    (base / "characterization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (base / "characterization.md").write_text(to_markdown(report), encoding="utf-8")
    return report
