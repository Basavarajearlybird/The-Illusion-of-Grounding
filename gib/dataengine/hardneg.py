"""Axis 1 -- retrieval noise (the hard-negative injector).

Random distractors are trivially ignored by strong models, so the benchmark's
difficulty lives entirely in the *hardness* of the negatives. Two stages produce
them:

1. **Semantic proximity.** Embed every ground-truth document, index them, and
   retrieve each pair's nearest neighbours *from other pairs*. These are on-topic
   and wrong -- the natural hard negative.
2. **Surgical factual corruption.** Mutate the pair's own ground-truth document:
   swap the predicate, the attribute, or the entity. A mutated ground-truth
   document is the hardest possible negative -- identical register, identical
   topic, one flipped fact.

The ground-truth document is *always* present and the ordering is shuffled with
the true document's position logged, which turns position bias into a free
ablation rather than an uncontrolled confound.

Every dose is a separate row keyed to the same ``pair_id``; that shared key is
what makes the decay-rate curve computable per pair rather than only in aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

from gib.dataengine.embed import Embedder, NeighbourIndex
from gib.dataengine.mutators import mutate_for_sub_type
from gib.schema import Axis, CleanPair, Variant, make_variant_id
from gib.seeds import rng_for


@dataclass
class HardNegativePool:
    """Per-pair distractor pool: mutated-self first, then semantic neighbours."""

    mutated: list[str]
    neighbours: list[str]
    injected_falsehood: str  # the fact the mutation asserts; powers taxonomy

    def take(self, n: int, hard_ratio: float, rng) -> list[str]:
        """Draw ``n`` distractors, front-loading the surgical ones.

        Surgical negatives are scarce (one per applicable mutation kind) and much
        harder, so they are spent first; semantic neighbours then pad out to the
        requested dose.
        """
        n_hard = min(len(self.mutated), max(1, round(n * hard_ratio)))
        picks = list(self.mutated[:n_hard])
        remaining = n - len(picks)
        if remaining > 0:
            pool = [d for d in self.neighbours if d not in picks]
            picks.extend(pool[:remaining])
        if len(picks) < n and self.neighbours:
            # Corpus too small to supply the full dose; recycle deterministically
            # rather than silently emitting an under-dosed row.
            while len(picks) < n:
                picks.append(rng.choice(self.neighbours))
        return picks[:n]


def build_pools(
    pairs: list[CleanPair],
    *,
    embedder_name: str = "BAAI/bge-large-en-v1.5",
    top_k: int = 32,
    live_wikidata: bool = False,
    seed_base: int = 20260721,
    verbose: bool = True,
) -> tuple[dict[str, HardNegativePool], dict[str, str]]:
    """Mine the distractor pool for every pair. Returns (pools, backend_info)."""
    docs = [p.ground_truth_doc for p in pairs]

    embedder = Embedder(embedder_name)
    vectors = embedder.encode(docs)
    index = NeighbourIndex(vectors)
    k = min(top_k + 1, len(pairs))  # +1: the query retrieves itself
    _, idx = index.search(vectors, k)

    info = {"embedder_backend": embedder.backend, "index_backend": index.backend,
            "embedder_model": embedder_name}
    if verbose:
        print(f"  [hardneg] embedder={embedder.backend} index={index.backend} k={k}")

    pools: dict[str, HardNegativePool] = {}
    for i, pair in enumerate(pairs):
        # -- stage 2: surgical corruption of this pair's own document -------- #
        mutated: list[str] = []
        falsehood = ""
        for kind_args in (
            dict(entity=pair.subject or pair.answer, attribute=pair.attribute, predicate=pair.predicate),
            dict(entity=pair.obj, attribute="", predicate=""),
        ):
            m = mutate_for_sub_type(
                pair.ground_truth_doc,
                pair.sub_type.value,
                key=pair.pair_id + str(len(mutated)),
                live=live_wikidata,
                **kind_args,
            )
            if m.applied and m.text not in mutated:
                mutated.append(m.text)
                if not falsehood:
                    falsehood = m.replacement

        # -- stage 1: semantic neighbours from *other* pairs ----------------- #
        neighbours = [docs[j] for j in idx[i] if j != i][:top_k]

        pools[pair.pair_id] = HardNegativePool(
            mutated=mutated, neighbours=neighbours, injected_falsehood=falsehood
        )

    return pools, info


def make_noise_variants(
    pair: CleanPair,
    pool: HardNegativePool,
    *,
    doses: list[int],
    hard_ratio: float = 0.5,
    seed_base: int = 20260721,
) -> list[Variant]:
    """One row per dose in {1, 3, 5, 10, 20}, ground truth always present."""
    out: list[Variant] = []
    for dose in doses:
        rng = rng_for("noise", pair.pair_id, dose, base=seed_base)
        distractors = pool.take(dose, hard_ratio, rng)

        docs = [pair.ground_truth_doc] + distractors
        order = list(range(len(docs)))
        rng.shuffle(order)
        shuffled = [docs[i] for i in order]
        gt_pos = order.index(0)

        out.append(
            Variant(
                pair_id=pair.pair_id,
                variant_id=make_variant_id(pair.pair_id, Axis.NOISE, dose),
                sub_type=pair.sub_type,
                axis=Axis.NOISE,
                dose=str(dose),
                image_path=pair.image_path,
                image_sha256=pair.image_sha256,
                question=pair.question,
                answer_visual=pair.answer,
                # The falsehood the hardest distractor asserts. Populated so a
                # model that parrots a distractor is classified as Knowledge
                # Inertia rather than lumped into Reasoning Collapse.
                answer_textual=pool.injected_falsehood,
                retrieved_docs=shuffled,
                doc_positions=order,
                gt_doc_position=gt_pos,
                bbox=pair.bbox,
                n_distractors=dose,
                source=pair.source,
                seed=seed_base,
            )
        )
    return out
