"""Axis 3 -- cross-modal conflict (the deception generator).

Take the clean image and rewrite its ground-truth document so the text asserts a
*different* entity, attribute, or relation than the image shows. Entity swaps go
through the WikiData sibling graph, so the falsehood is same-category and
plausible ("apple" -> "orange", never "apple" -> "helicopter"); attribute and
predicate swaps go through their family maps for the same reason.

**Counterfactual twins.** The flipped case is built into the *dataset*, not into
the evaluation logic. The architecture's argument for this is worth restating
because it drives the implementation below: building the flip into the data
guarantees the flip is exact, and it lets any downstream user of GIB-15K run
counterfactual probing for free.

Constructing the flip needs care on real imagery. Given an item whose image shows
A and whose text asserts B, the true counterfactual needs an image showing B with
text asserting A -- and no amount of text rewriting can produce an image of an
orange from a photograph of an apple. So the twin is *mined from the corpus*: a
real pair of the same sub-type whose visual answer is B, injected with the
falsehood A. Both rows then share a ``twin_id``.

Items for which no partner exists are still emitted (they are valid conflict
items) but carry an empty ``twin_id``, and the counterfactual-symmetry analysis
in Layer 5 uses only twinned rows. Reporting the twinned subset rather than
quietly treating unpaired items as twins is what keeps the symmetry score honest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from gib.dataengine.mutators import Mutation, mutate_for_sub_type
from gib.schema import Axis, CleanPair, SubType, Variant, make_variant_id
from gib.seeds import rng_for


@dataclass
class ConflictSpec:
    """A planned conflict: which pair, which falsehood, which twin."""

    pair: CleanPair
    mutation: Mutation
    twin_id: str = ""


def _target_fact(pair: CleanPair) -> str:
    """The fact the question asks about -- the thing the mutation must flip."""
    if pair.sub_type is SubType.ATTRIBUTE:
        return pair.attribute or pair.answer
    if pair.sub_type is SubType.RELATION:
        return pair.predicate or pair.answer
    return pair.answer


def plan_conflicts(
    pairs: list[CleanPair],
    *,
    make_twins: bool = True,
    live_wikidata: bool = False,
    seed_base: int = 20260721,
) -> list[ConflictSpec]:
    """Mutate each pair's document, then mine counterfactual partners."""
    specs: list[ConflictSpec] = []
    for pair in pairs:
        m = mutate_for_sub_type(
            pair.ground_truth_doc,
            pair.sub_type.value,
            entity=pair.subject or pair.answer,
            attribute=pair.attribute,
            predicate=pair.predicate,
            key=f"conflict:{pair.pair_id}",
            live=live_wikidata,
        )
        if not m.applied:
            # No principled swap exists for this fact -- skip rather than invent
            # an implausible one, which would make the item easy for the wrong
            # reason.
            continue
        specs.append(ConflictSpec(pair=pair, mutation=m))

    if not make_twins:
        return specs

    # -- mine twins ---------------------------------------------------------- #
    # Index by (sub_type, the fact the image shows) so a partner can be found
    # for the falsehood we injected.
    by_fact: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, spec in enumerate(specs):
        by_fact[(spec.pair.sub_type.value, _target_fact(spec.pair))].append(i)

    used: set[int] = set()
    for i, spec in enumerate(specs):
        if i in used or spec.twin_id:
            continue
        want = (spec.pair.sub_type.value, spec.mutation.replacement)
        rng = rng_for("twin", spec.pair.pair_id, base=seed_base)

        partners = [
            j
            for j in by_fact.get(want, [])
            if j != i
            and j not in used
            and not specs[j].twin_id
            # A true counterfactual requires the partner's falsehood to be the
            # fact *this* item's image shows -- i.e. the flip is exact.
            and specs[j].mutation.replacement == _target_fact(spec.pair)
            and specs[j].pair.image_id != spec.pair.image_id
        ]
        if not partners:
            continue
        j = rng.choice(partners)
        twin_id = f"twin-{min(spec.pair.pair_id, specs[j].pair.pair_id)}"
        specs[i].twin_id = twin_id
        specs[j].twin_id = twin_id
        used.update({i, j})

    return specs


def make_conflict_variant(spec: ConflictSpec, seed_base: int = 20260721) -> Variant:
    """One conflict row: clean image, poisoned document.

    The document set contains *only* the mutated passage. Including the truthful
    document alongside it would turn the item into a retrieval-noise item with a
    distractor, which is Axis 1's job. Axis 3 is the pure clash: everything the
    text says contradicts what the image shows.
    """
    pair = spec.pair
    return Variant(
        pair_id=pair.pair_id,
        variant_id=make_variant_id(pair.pair_id, Axis.CONFLICT, "na"),
        sub_type=pair.sub_type,
        axis=Axis.CONFLICT,
        dose="n/a",
        image_path=pair.image_path,
        image_sha256=pair.image_sha256,
        question=pair.question,
        answer_visual=pair.answer,
        answer_textual=spec.mutation.replacement,
        retrieved_docs=[spec.mutation.text],
        doc_positions=[0],
        gt_doc_position=-1,  # the ground-truth document is absent by design
        twin_id=spec.twin_id,
        bbox=pair.bbox,
        source=pair.source,
        seed=seed_base,
    )


def make_clean_variant(pair: CleanPair, seed_base: int = 20260721) -> Variant:
    """The control condition: clean image, correct document.

    Every robustness quantity in the paper is a ratio against this row -- the
    Robustness Score's denominator, the decay curve's intercept, the heatmap's
    delta -- so the clean config is generated for every pair, not sampled.
    """
    return Variant(
        pair_id=pair.pair_id,
        variant_id=make_variant_id(pair.pair_id, Axis.CLEAN, "na"),
        sub_type=pair.sub_type,
        axis=Axis.CLEAN,
        dose="n/a",
        image_path=pair.image_path,
        image_sha256=pair.image_sha256,
        question=pair.question,
        answer_visual=pair.answer,
        answer_textual="",
        retrieved_docs=[pair.ground_truth_doc],
        doc_positions=[0],
        gt_doc_position=0,
        bbox=pair.bbox,
        source=pair.source,
        seed=seed_base,
    )
