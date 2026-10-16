"""The automatic failure taxonomy classifier (Layer 2.4).

Because the schema carries both ``answer_visual`` and ``answer_textual``, the
taxonomy stops being a qualitative label applied to a hand-coded subset and
becomes a deterministic function of two judge verdicts:

===========================  ====================================================
Judge result                 Classification
===========================  ====================================================
matches ``answer_visual``    **Correct** (grounded)
matches ``answer_textual``   **Knowledge Inertia** -- ignored the image, followed
                             the poisoned text
wrong, consistent with the   **Sensory Capture** -- followed a degraded percept
degraded percept             over correct text
matches neither, incoherent  **Reasoning Collapse**
===========================  ====================================================

This is what makes taxonomy proportions reportable across the full 60K corpus
instead of a hand-labelled sample, and it should be claimed in the paper as a
design contribution rather than an implementation detail.

The one genuinely inferential row is Sensory Capture, and it is treated
carefully. On the visual axis the retrieved text is *correct*, so a wrong answer
means the model preferred a degraded percept to accurate text -- which is the
definition. Off the visual axis there is no degraded percept to be captured by,
so the label is not available and the row falls to Reasoning Collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

from gib.eval.scoring import is_refusal
from gib.schema import Axis, FailureMode


@dataclass
class TaxonomyInput:
    axis: str
    matches_visual: bool
    matches_textual: bool
    prediction: str
    answer_textual: str = ""
    degradation: str = "none"


def classify(inp: TaxonomyInput) -> FailureMode:
    if inp.matches_visual:
        return FailureMode.CORRECT

    # Knowledge Inertia requires an actual injected falsehood to have been
    # followed. Without `answer_textual` the label would be unfalsifiable.
    if inp.matches_textual and inp.answer_textual:
        return FailureMode.KNOWLEDGE_INERTIA

    if inp.axis in (Axis.VISUAL.value, Axis.COMBINED.value) and inp.degradation != "none":
        # A refusal under degradation is not Sensory Capture -- the model
        # declined to guess, which is the opposite failure.
        if not is_refusal(inp.prediction):
            return FailureMode.SENSORY_CAPTURE

    return FailureMode.REASONING_COLLAPSE


def classify_row(row: dict) -> str:
    return classify(
        TaxonomyInput(
            axis=row.get("axis", "clean"),
            matches_visual=bool(row.get("matches_visual")),
            matches_textual=bool(row.get("matches_textual")),
            prediction=row.get("answer", ""),
            answer_textual=row.get("answer_textual", ""),
            degradation=row.get("degradation", "none"),
        )
    ).value


TAXONOMY_DEFINITIONS = {
    FailureMode.CORRECT.value: (
        "Grounded. The answer matches what the image shows."
    ),
    FailureMode.KNOWLEDGE_INERTIA.value: (
        "The model ignored the image and followed incorrect retrieved text, "
        "reproducing the injected falsehood verbatim. This is the Illusion of "
        "Grounding in its purest form: the output is fluent, sourced, and wrong."
    ),
    FailureMode.SENSORY_CAPTURE.value: (
        "The model ignored correct retrieved text and committed to a reading of "
        "a degraded image. The percept overrode the document."
    ),
    FailureMode.REASONING_COLLAPSE.value: (
        "The model matched neither modality, producing an off-topic, incoherent, "
        "or self-contradictory answer under the stressor."
    ),
}
