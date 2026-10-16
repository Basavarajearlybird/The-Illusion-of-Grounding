"""Structural mutators: predicate, attribute, and entity swaps.

Shared by Axis 1 (surgical hard negatives) and Axis 3 (cross-modal conflict).
Both axes need the same operation -- produce a passage identical to the truth
except for exactly one flipped fact -- and differ only in how the result is used:
Axis 1 injects it as a distractor alongside the true document, Axis 3 *replaces*
the true document with it.

Mutation is structural. The document generator records which span carries the
fact, so a swap is a targeted replacement rather than a global string substitution
that could clobber an incidental occurrence of the same word.

spaCy is used when installed (better sentence handling on LLM-paraphrased
documents); the deterministic rule-based path is the default so the build has no
model download in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gib.dataengine.sources.visual_genome import (
    ATTRIBUTE_FAMILIES,
    ATTRIBUTE_TO_FAMILY,
    PREDICATE_CANON,
)
from gib.dataengine.sources.wikidata import default_graph
from gib.seeds import rng_for

# Predicates grouped so a swap stays type-plausible: an action swaps for an
# action, a spatial relation for a spatial relation. Swapping "holding" for
# "above" would produce a distractor that is odd rather than deceptive.
PREDICATE_FAMILIES: dict[str, list[str]] = {
    "manipulation": ["holding", "carrying", "throwing", "using", "touching", "pushing", "pulling"],
    "posture": ["sitting on", "standing on", "lying on"],
    "transport": ["riding", "driving", "pushing"],
    "consumption": ["eating", "drinking"],
    "attire": ["wearing", "carrying"],
    "attention": ["looking at", "playing with", "using"],
    "spatial": [
        "on", "in", "under", "above", "behind", "in front of", "near",
        "next to", "on top of", "to the left of", "to the right of", "below",
    ],
}
# Membership is genuinely set-valued: "pushing" is both a manipulation and a
# transport predicate. A single-valued map would silently keep only whichever
# family was declared last, which both narrows the candidate pool and makes the
# reverse lookup wrong.
PREDICATE_TO_FAMILIES: dict[str, list[str]] = {}
for _fam, _members in PREDICATE_FAMILIES.items():
    for _p in _members:
        PREDICATE_TO_FAMILIES.setdefault(_p, []).append(_fam)


@dataclass
class Mutation:
    """The result of one surgical swap."""

    text: str
    kind: str  # predicate | attribute | entity
    original: str
    replacement: str

    @property
    def applied(self) -> bool:
        return bool(self.replacement) and self.replacement != self.original


def _replace_span(text: str, original: str, replacement: str) -> str:
    """Replace the first whole-word occurrence, preserving leading capitalisation."""
    pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return text
    found = m.group(0)
    repl = replacement.capitalize() if found[:1].isupper() else replacement
    return text[: m.start()] + repl + text[m.end() :]


# --------------------------------------------------------------------------- #
# The three swaps
# --------------------------------------------------------------------------- #


def swap_predicate(text: str, predicate: str, key: str = "") -> Mutation:
    """"holding" -> "throwing". Stays within the predicate's family."""
    p = PREDICATE_CANON.get(predicate.strip().lower(), predicate.strip().lower())
    fams = PREDICATE_TO_FAMILIES.get(p)
    if not fams:
        return Mutation(text, "predicate", p, "")
    # Draw from every family the predicate belongs to, so a predicate with two
    # senses keeps both pools of plausible substitutes.
    candidates = sorted({c for fam in fams for c in PREDICATE_FAMILIES[fam] if c != p})
    if not candidates:
        return Mutation(text, "predicate", p, "")
    rng = rng_for("swap_predicate", key or text, p)
    repl = rng.choice(candidates)
    return Mutation(_replace_span(text, p, repl), "predicate", p, repl)


def swap_attribute(text: str, attribute: str, key: str = "") -> Mutation:
    """"red" -> "green". Stays within the attribute's family, so the result is
    guaranteed to be a false assertion rather than a compatible one (a thing can
    be both "red" and "wooden", but not both "red" and "green")."""
    a = attribute.strip().lower()
    fam = ATTRIBUTE_TO_FAMILY.get(a)
    if not fam:
        return Mutation(text, "attribute", a, "")
    candidates = [c for c in ATTRIBUTE_FAMILIES[fam] if c != a]
    if not candidates:
        return Mutation(text, "attribute", a, "")
    rng = rng_for("swap_attribute", key or text, a)
    repl = rng.choice(candidates)
    return Mutation(_replace_span(text, a, repl), "attribute", a, repl)


def swap_entity(text: str, entity: str, key: str = "", live: bool = False) -> Mutation:
    """"apple" -> "orange", via the WikiData sibling graph."""
    e = entity.strip().lower()
    sibs = default_graph(live=live).siblings(e)
    if not sibs:
        return Mutation(text, "entity", e, "")
    rng = rng_for("swap_entity", key or text, e)
    repl = rng.choice(sibs)
    return Mutation(_replace_span(text, e, repl), "entity", e, repl)


def mutate_for_sub_type(
    text: str,
    sub_type: str,
    *,
    entity: str = "",
    attribute: str = "",
    predicate: str = "",
    key: str = "",
    live: bool = False,
) -> Mutation:
    """Dispatch to the swap that flips the fact the question actually asks about.

    This alignment is essential: for an attribute question, corrupting the entity
    would leave the *answer* still correct, and the item would silently stop
    testing what it claims to test.
    """
    st = str(sub_type)
    if st == "relation" and predicate:
        m = swap_predicate(text, predicate, key)
        if m.applied:
            return m
    if st == "attribute" and attribute:
        m = swap_attribute(text, attribute, key)
        if m.applied:
            return m
    if entity:
        m = swap_entity(text, entity, key, live=live)
        if m.applied:
            return m
    # Fall back through the remaining swaps so a mutation is essentially always
    # produced; the caller checks `.applied` and drops the item if not.
    for cand in (
        swap_attribute(text, attribute, key) if attribute else None,
        swap_predicate(text, predicate, key) if predicate else None,
    ):
        if cand is not None and cand.applied:
            return cand
    return Mutation(text, "none", "", "")
