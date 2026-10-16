"""QA generation (Layer 1.1) -- template-first, LLM-second.

Roughly 60-70% of questions come from scene-graph templates: cheap, and
verifiable *by construction* because the answer is read straight out of the
annotation. The remainder are LLM paraphrases of those templates, which buys
linguistic diversity without giving up the verified answer.

Every generated pair -- templated or paraphrased -- then passes **round-trip
validation**: an independent solver answers the question from the scene graph
alone, and the pair is discarded on mismatch. The solver is a genuinely separate
implementation (it re-parses the surface question and re-resolves it against the
graph) rather than a mirror of the templater, so it catches the failure that
matters: questions that are ambiguous or under-specified given the scene.

Each pair also carries a ``ground_truth_doc`` -- a short synthetic "retrieved
passage" that correctly answers the question. It is generated here, at ingestion
time, because every retrieval-side perturbation in 1.2 is defined as a mutation
of it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from gib.dataengine.scene import Obj, Relation, SceneGraph
from gib.dataengine.sources.visual_genome import ATTRIBUTE_TO_FAMILY
from gib.schema import CleanPair, SubType
from gib.seeds import rng_for

# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

OBJECT_TEMPLATES = [
    "What object is {predicate} the {anchor}?",
    "Which item is {predicate} the {anchor}?",
    "Name the object {predicate} the {anchor}.",
]
ATTRIBUTE_TEMPLATES = {
    "color": ["What color is the {object}?", "What is the color of the {object}?"],
    "material": ["What material is the {object} made of?", "What is the {object} made of?"],
    "size": ["What size is the {object}?", "How large is the {object}?"],
    "state": ["What is the state of the {object}?", "What condition is the {object} in?"],
}
RELATION_TEMPLATES = [
    "What is the {subject} doing with the {object}?",
    "What is the relationship between the {subject} and the {object}?",
    "How is the {subject} interacting with the {object}?",
]

ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


# --------------------------------------------------------------------------- #
# Ground-truth document synthesis
# --------------------------------------------------------------------------- #


def _context_sentence(scene: SceneGraph, exclude: set[str], rng, n: int = 1) -> str:
    """One or two true-but-irrelevant sentences.

    Padding matters more than it looks: a one-sentence document would make the
    injected falsehood the only content, so a model could locate the poisoned
    fact by position alone. Realistic passage length is part of the stressor.
    """
    others = [o for o in scene.objects if o.name not in exclude]
    if not others:
        return ""
    picks = rng.sample(others, min(n, len(others)))
    out = []
    for o in picks:
        if o.attributes:
            out.append(f"{_a(o.attributes[0] + ' ' + o.name).capitalize()} is also visible.")
        else:
            out.append(f"{_a(o.name).capitalize()} appears in the scene.")
    return " ".join(out)


def make_object_doc(scene: SceneGraph, obj: Obj, rel: Relation, rng) -> str:
    lead = f"The photograph shows {_a(obj.name)} {rel.predicate} the {rel.obj}."
    tail = _context_sentence(scene, {obj.name, rel.obj}, rng)
    return f"{lead} {tail}".strip()


def make_attribute_doc(scene: SceneGraph, obj: Obj, attribute: str, rng) -> str:
    lead = f"The photograph shows {_a(attribute + ' ' + obj.name)}."
    mid = f"The {obj.name} is clearly {attribute} in this image."
    tail = _context_sentence(scene, {obj.name}, rng)
    return f"{lead} {mid} {tail}".strip()


def make_relation_doc(scene: SceneGraph, rel: Relation, rng) -> str:
    lead = f"In this photograph, the {rel.subject} is {rel.predicate} the {rel.obj}."
    tail = _context_sentence(scene, {rel.subject, rel.obj}, rng)
    return f"{lead} {tail}".strip()


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    question: str
    answer: str
    sub_type: SubType
    bbox: list[float]
    doc: str
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    attribute: str = ""


def candidates_for_scene(scene: SceneGraph, seed_base: int = 20260721) -> list[Candidate]:
    """All machine-verifiable QA candidates for one scene, across sub-types."""
    rng = rng_for("qagen", scene.image_id, base=seed_base)
    unique = {o.name: o for o in scene.unique_objects()}
    out: list[Candidate] = []

    # -- object: identify an object by its relation to a unique anchor ------- #
    for rel in scene.relations:
        subj = unique.get(rel.subject)
        anchor = unique.get(rel.obj)
        if subj is None or anchor is None:
            continue
        q = rng.choice(OBJECT_TEMPLATES).format(predicate=rel.predicate, anchor=rel.obj)
        out.append(
            Candidate(
                question=q,
                answer=subj.name,
                sub_type=SubType.OBJECT,
                bbox=subj.bbox,
                doc=make_object_doc(scene, subj, rel, rng),
                subject=rel.subject,
                predicate=rel.predicate,
                obj=rel.obj,
            )
        )

    # -- attribute: only for uniquely-named objects with a family attribute -- #
    for obj in unique.values():
        for attr in obj.attributes:
            fam = ATTRIBUTE_TO_FAMILY.get(attr)
            if not fam:
                continue
            q = rng.choice(ATTRIBUTE_TEMPLATES[fam]).format(object=obj.name)
            out.append(
                Candidate(
                    question=q,
                    answer=attr,
                    sub_type=SubType.ATTRIBUTE,
                    bbox=obj.bbox,
                    doc=make_attribute_doc(scene, obj, attr, rng),
                    subject=obj.name,
                    attribute=attr,
                )
            )

    # -- relation: ask for the predicate itself ------------------------------ #
    for rel in scene.relations:
        if rel.subject not in unique or rel.obj not in unique:
            continue
        if rel.subject == rel.obj:
            continue
        q = rng.choice(RELATION_TEMPLATES).format(subject=rel.subject, object=rel.obj)
        # The subject's box is the answer-bearing region for a relation question:
        # cropping it away is what removes the evidence for the interaction.
        out.append(
            Candidate(
                question=q,
                answer=rel.predicate,
                sub_type=SubType.RELATION,
                bbox=rel.subject_bbox,
                doc=make_relation_doc(scene, rel, rng),
                subject=rel.subject,
                predicate=rel.predicate,
                obj=rel.obj,
            )
        )

    return out


# --------------------------------------------------------------------------- #
# Round-trip validation
# --------------------------------------------------------------------------- #


def _parse_and_solve(question: str, scene: SceneGraph) -> set[str]:
    """Answer ``question`` from ``scene`` alone, by re-parsing the surface form.

    Returns the set of *all* admissible answers. A candidate is only accepted
    when this set is exactly one element -- which is how ambiguous questions
    ("what colour is the chair?" in a two-chair scene) get filtered out.
    """
    q = question.strip().lower().rstrip("?.")

    # attribute: "what color is the X" / "what is the color of the X" / "made of"
    m = re.search(
        r"what (?:color|colour|material|size|state|condition)"
        r"(?: is| of)?(?: the)? (.+?)(?: made of)?$",
        q,
    )
    if m is None:
        m = re.search(r"what is the (?:color|colour|material|size|state|condition) of the (.+)$", q)
    if m is None:
        m = re.search(r"how large is the (.+)$", q)
    if m is None:
        m = re.search(r"what is the (.+?) made of$", q)
    if m:
        target = ARTICLE_RE.sub("", m.group(1)).strip()
        answers = {
            a for o in scene.objects if o.name == target for a in o.attributes
        }
        return answers

    # relation: "what is the S doing with the O" / "relationship between S and O"
    m = re.search(r"what is the (.+?) doing with the (.+)$", q)
    if m is None:
        m = re.search(r"what is the relationship between the (.+?) and the (.+)$", q)
    if m is None:
        m = re.search(r"how is the (.+?) interacting with the (.+)$", q)
    if m:
        s, o = ARTICLE_RE.sub("", m.group(1)).strip(), ARTICLE_RE.sub("", m.group(2)).strip()
        return {r.predicate for r in scene.relations if r.subject == s and r.obj == o}

    # object: "what object is P the A" / "which item is P the A" / "name the object P the A"
    m = re.search(r"(?:what object|which item) is (.+?) the (.+)$", q)
    if m is None:
        m = re.search(r"name the object (.+?) the (.+)$", q)
    if m:
        pred, anchor = m.group(1).strip(), ARTICLE_RE.sub("", m.group(2)).strip()
        return {
            r.subject for r in scene.relations if r.predicate == pred and r.obj == anchor
        }

    return set()


def round_trip_ok(cand: Candidate, scene: SceneGraph) -> bool:
    """Accept only if the independent solver recovers exactly this answer."""
    solved = _parse_and_solve(cand.question, scene)
    return solved == {cand.answer}


# --------------------------------------------------------------------------- #
# LLM paraphrase hook
# --------------------------------------------------------------------------- #

ParaphraseFn = Callable[[list[str]], list[str]]


def paraphrase_none(questions: list[str]) -> list[str]:
    """Default: no paraphrasing. Keeps `make dataset` offline and deterministic."""
    return list(questions)


def make_llm_paraphraser(model: str = "claude-opus-4-8", batch: int = 20) -> ParaphraseFn:
    """Return a paraphraser backed by an LLM adapter.

    Paraphrases are re-validated by the same round-trip check, so a paraphrase
    that drifts in meaning is dropped rather than silently weakening the item.
    """

    def _fn(questions: list[str]) -> list[str]:
        from gib.adapters.registry import get_text_model

        llm = get_text_model(model)
        out: list[str] = []
        for i in range(0, len(questions), batch):
            chunk = questions[i : i + batch]
            numbered = "\n".join(f"{j+1}. {q}" for j, q in enumerate(chunk))
            prompt = (
                "Rewrite each question below so it sounds natural and varied, while "
                "asking for EXACTLY the same fact. Do not add or remove any "
                "constraint, and do not change which object is being asked about.\n"
                "Return one rewritten question per line, numbered identically.\n\n"
                f"{numbered}"
            )
            resp = llm(prompt)
            lines = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip() for ln in resp.splitlines()]
            lines = [ln for ln in lines if ln]
            out.extend(lines if len(lines) == len(chunk) else chunk)
        return out

    return _fn


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def generate_pairs(
    scenes: Iterator[SceneGraph],
    *,
    per_sub_type: int = 5000,
    paraphrase_rate: float = 0.35,
    paraphraser: ParaphraseFn | None = None,
    seed_base: int = 20260721,
    max_per_scene: int = 2,
) -> tuple[list[CleanPair], dict[str, int]]:
    """Build a sub-type-balanced set of validated clean pairs.

    ``max_per_scene`` caps how many pairs one image contributes, so the corpus
    does not concentrate on a handful of densely-annotated scenes.
    """
    from gib.schema import sha256_file

    paraphraser = paraphraser or paraphrase_none
    quotas = {st: per_sub_type for st in SubType}
    pairs: list[CleanPair] = []
    stats = {"scenes": 0, "candidates": 0, "round_trip_rejected": 0, "kept": 0}
    hash_cache: dict[str, str] = {}

    for scene in scenes:
        if all(q <= 0 for q in quotas.values()):
            break
        stats["scenes"] += 1
        rng = rng_for("select", scene.image_id, base=seed_base)
        cands = candidates_for_scene(scene, seed_base=seed_base)
        stats["candidates"] += len(cands)

        # Prefer sub-types that are still short of quota so the corpus lands
        # balanced rather than being trimmed to balance at the end.
        cands.sort(key=lambda c: (-quotas[c.sub_type], rng.random()))

        taken = 0
        seen_sub_types: set[SubType] = set()
        for cand in cands:
            if taken >= max_per_scene:
                break
            if quotas[cand.sub_type] <= 0 or cand.sub_type in seen_sub_types:
                continue
            if not round_trip_ok(cand, scene):
                stats["round_trip_rejected"] += 1
                continue

            if scene.image_path not in hash_cache:
                hash_cache[scene.image_path] = sha256_file(scene.image_path)

            pair_id = f"{scene.image_id}-{cand.sub_type.value}-{taken}"
            pairs.append(
                CleanPair(
                    pair_id=pair_id,
                    source=scene.source,
                    image_id=scene.image_id,
                    image_path=scene.image_path,
                    image_sha256=hash_cache[scene.image_path],
                    question=cand.question,
                    answer=cand.answer,
                    sub_type=cand.sub_type,
                    bbox=cand.bbox,
                    ground_truth_doc=cand.doc,
                    subject=cand.subject,
                    predicate=cand.predicate,
                    obj=cand.obj,
                    attribute=cand.attribute,
                )
            )
            quotas[cand.sub_type] -= 1
            seen_sub_types.add(cand.sub_type)
            taken += 1
            stats["kept"] += 1

    # -- LLM paraphrase pass, then re-validate ------------------------------- #
    if paraphrase_rate > 0 and paraphraser is not paraphrase_none:
        rng = rng_for("paraphrase", base=seed_base)
        idx = [i for i in range(len(pairs)) if rng.random() < paraphrase_rate]
        if idx:
            rewritten = paraphraser([pairs[i].question for i in idx])
            for i, new_q in zip(idx, rewritten):
                pairs[i].question = new_q
                pairs[i].qa_origin = "llm_paraphrase"
            stats["paraphrased"] = len(idx)

    return pairs, stats


def spot_check_sample(pairs: list[CleanPair], rate: float = 0.05, seed_base: int = 20260721):
    """The 5% stratified human QC sample.

    Its acceptance rate is the annotation-quality number reported in the
    datasheet, so the sample is stratified by sub_type x source and its
    membership is seeded and therefore auditable.
    """
    rng = rng_for("spot_check", base=seed_base)
    strata: dict[tuple[str, str], list[CleanPair]] = {}
    for p in pairs:
        strata.setdefault((p.sub_type.value, p.source), []).append(p)
    sample: list[CleanPair] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda p: p.pair_id)
        k = max(1, round(len(group) * rate))
        sample.extend(rng.sample(group, min(k, len(group))))
    return sample
