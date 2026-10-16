"""Simulated participant responses -- pipeline verification ONLY.

=============================================================================
 THESE ARE NOT PEOPLE. Responses written by this module carry participant ids
 prefixed ``sim-`` and land in ``responses_simulated.jsonl``, never in the
 real ``responses.jsonl``. No number derived from them belongs in the paper.
=============================================================================

Layer 4 otherwise cannot be exercised without recruiting participants, which
would leave the scoring path, Krippendorff's alpha, and the human-gap figure
untested until the very end of the project. This module closes that gap.

The behavioural model encodes the study's actual hypothesis so the analysis code
is exercised against data with the right shape: simulated participants are near
ceiling on cross-modal conflict (people trivially trust their eyes over a
contradicting paragraph) and degrade sharply under extreme blur and tight crops
(where human perception genuinely fails too). The narrowing case is modelled on
purpose -- an analysis path that only ever sees a flattering gap is not tested.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gib.config import HUMAN_DIR
from gib.human.sample import HumanItem
from gib.seeds import rng_for

SIM_RESPONSES_PATH = HUMAN_DIR / "responses_simulated.jsonl"


def _p_correct(item: HumanItem, skill: float) -> float:
    """Human accuracy model, per stressor."""
    axis = item.axis

    if axis == "clean":
        p = 0.97
    elif axis == "conflict":
        # The paper's central asymmetry: humans are near-ceiling here, which is
        # exactly where models collapse.
        p = 0.94
    elif axis == "noise":
        # Distractor documents barely affect a human who can just look.
        n = float(item.dose) if item.dose.isdigit() else 0.0
        p = 0.95 - 0.004 * n
    elif axis == "visual":
        d = item.dose
        if d.startswith("sigma"):
            sigma = float(d.replace("sigma", ""))
            p = 0.96 / (1.0 + (sigma / 7.0) ** 2.4)      # honest collapse at high σ
        elif d.startswith("kernel"):
            k = float(d.replace("kernel", ""))
            p = 0.95 / (1.0 + (k / 28.0) ** 2.0)
        elif d.startswith("crop"):
            pct = float(d.replace("crop", "")) / 100.0
            p = 0.93 / (1.0 + (0.09 / max(pct, 1e-3)) ** 1.4)
        else:
            p = 0.9
    else:  # combined
        p = 0.72

    return max(0.02, min(0.995, p * skill))


def _response_time(item: HumanItem, rng) -> float:
    """Longer under harder stressors, with a lognormal-ish spread."""
    base = {"clean": 6.0, "noise": 11.0, "visual": 13.0,
            "conflict": 15.0, "combined": 18.0}.get(item.axis, 10.0)
    base += 1.6 * len(item.retrieved_docs)
    return round(max(1.5, base * rng.lognormvariate(0.0, 0.35)), 2)


def simulate_responses(
    items: list[HumanItem],
    *,
    n_participants: int = 8,
    items_per_participant: int | None = None,
    seed: int = 20260721,
    path: Path | None = None,
) -> Path:
    """Write simulated responses for the whole study sample.

    Each participant sees an overlapping subset rather than everything, which is
    what makes the missing-data case real and is precisely why the agreement
    statistic is Krippendorff's alpha rather than Fleiss' kappa.
    """
    path = Path(path or SIM_RESPONSES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    per = items_per_participant or max(1, int(len(items) * 0.6))
    lines: list[str] = []

    for pi in range(n_participants):
        pid = f"sim-p{pi:02d}"
        prng = rng_for("human_sim_participant", pid, base=seed)
        # Per-participant skill multiplier: real inter-rater variance, which the
        # agreement statistic needs in order to be non-degenerate.
        skill = prng.uniform(0.93, 1.03)
        subset = prng.sample(items, min(per, len(items)))

        for item in subset:
            rng = rng_for("human_sim", pid, item.variant_id, base=seed)
            p = _p_correct(item, skill)
            if rng.random() < p:
                answer = item.answer_visual
            elif item.answer_textual and rng.random() < 0.18:
                # Humans are not immune to a confident passage -- just far more
                # resistant than models.
                answer = item.answer_textual
            else:
                answer = "unsure"
            lines.append(
                json.dumps(
                    {
                        "participant_id": pid,
                        "variant_id": item.variant_id,
                        "answer": answer,
                        "response_time_s": _response_time(item, rng),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "simulated": True,
                    }
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[human] *** SIMULATED *** {len(lines)} responses from {n_participants} "
          f"synthetic participants -> {path}")
    print("[human] These are not people. Do not report numbers derived from them.")
    return path
