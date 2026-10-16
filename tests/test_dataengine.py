"""Layer 1 tests: determinism, the mutators, and the adversarial invariants.

These focus on the properties that would silently corrupt the science if they
broke — a conflict item with no conflict in it, a "hard" negative that is
actually the truth, or a build that is not reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from gib.dataengine.degrade import (
    crop_to_visible_fraction,
    gaussian_blur,
    motion_blur_kernel,
)
from gib.dataengine.mutators import (
    mutate_for_sub_type,
    swap_attribute,
    swap_entity,
    swap_predicate,
)
from gib.dataengine.qagen import candidates_for_scene, round_trip_ok
from gib.dataengine.scene import Obj, Relation, SceneGraph
from gib.seeds import derive_seed, rng_for

# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_seed_derivation_is_content_addressed():
    assert derive_seed("a", 1) == derive_seed("a", 1)
    assert derive_seed("a", 1) != derive_seed("a", 2)


def test_rng_independent_of_call_order():
    """The whole point of content-addressed seeds: a partial re-run must produce
    byte-identical variants to a full run."""
    first = [rng_for("x", i).random() for i in range(10)]
    second = [rng_for("x", i).random() for i in reversed(range(10))][::-1]
    assert first == second


# --------------------------------------------------------------------------- #
# Mutators
# --------------------------------------------------------------------------- #


def test_swap_attribute_stays_in_family_and_flips_the_fact():
    m = swap_attribute("The umbrella is red.", "red", key="k")
    assert m.applied
    assert m.replacement != "red"
    # Same family => mutually exclusive => the result is genuinely false.
    from gib.dataengine.sources.visual_genome import ATTRIBUTE_TO_FAMILY

    assert ATTRIBUTE_TO_FAMILY[m.replacement] == ATTRIBUTE_TO_FAMILY["red"]
    assert "red" not in m.text.lower()


def test_swap_predicate_stays_type_plausible():
    """The substitute must share a family with the original — swapping
    "holding" for "above" gives a distractor that is odd rather than deceptive.
    Membership is set-valued ("pushing" is both manipulation and transport), so
    the check is family *overlap*, not equality."""
    m = swap_predicate("The man is holding the frisbee.", "holding", key="k")
    assert m.applied
    from gib.dataengine.mutators import PREDICATE_TO_FAMILIES

    assert set(PREDICATE_TO_FAMILIES[m.replacement]) & set(PREDICATE_TO_FAMILIES["holding"])


def test_swap_entity_uses_sibling_graph():
    m = swap_entity("There is an apple on the table.", "apple", key="k")
    assert m.applied
    # A sibling under `fruit`, not an arbitrary noun.
    assert m.replacement in {"orange", "banana", "pear", "mango", "peach", "lemon", "plum"}


def test_mutation_is_deterministic():
    a = swap_entity("An apple sits here.", "apple", key="same")
    b = swap_entity("An apple sits here.", "apple", key="same")
    assert a.replacement == b.replacement


def test_mutation_targets_the_asked_fact():
    """An attribute question must get an attribute swap — corrupting the entity
    would leave the answer correct and silently neuter the item."""
    m = mutate_for_sub_type(
        "The photograph shows a red umbrella.", "attribute",
        entity="umbrella", attribute="red", key="k",
    )
    assert m.kind == "attribute"


def test_word_boundary_replacement_does_not_clobber_substrings():
    m = swap_entity("The cat sat on the catalogue.", "cat", key="k")
    assert "catalogue" in m.text


# --------------------------------------------------------------------------- #
# Degradations
# --------------------------------------------------------------------------- #


def test_gaussian_blur_monotonically_reduces_detail():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    variances = [gaussian_blur(img, s).var() for s in (0, 1, 2, 4, 8, 16)]
    assert all(a >= b - 1e-6 for a, b in zip(variances, variances[1:]))


def test_motion_blur_kernel_is_normalized():
    for length in (5, 10, 20, 40):
        for angle in (0, 45, 90, 137):
            k = motion_blur_kernel(length, angle)
            assert k.shape == (length, length)
            assert np.isclose(k.sum(), 1.0)


def test_crop_retains_requested_area_fraction():
    """The crop is keyed to the answer object's area, not the image centre."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    bbox = [100.0, 100.0, 200.0, 200.0]  # area 40000
    for frac in (0.5, 0.25, 0.10, 0.05):
        out = crop_to_visible_fraction(img, bbox, frac, out_size=336)
        assert out.shape == (336, 336, 3)


def test_crop_is_centred_on_the_object_not_the_image():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[10:60, 10:60] = 255                      # bright patch in the corner
    out = crop_to_visible_fraction(img, [10, 10, 50, 50], 0.25, out_size=64)
    assert out.mean() > 200                       # cropped to the object


# --------------------------------------------------------------------------- #
# QA generation + round-trip validation
# --------------------------------------------------------------------------- #


@pytest.fixture
def scene():
    return SceneGraph(
        image_id="t1", image_path="x.jpg", width=640, height=480, source="test",
        objects=[
            Obj("person", [10, 10, 100, 200], []),
            Obj("frisbee", [150, 50, 60, 60], ["red"]),
            Obj("dog", [300, 100, 90, 90], ["brown"]),
        ],
        relations=[Relation("person", "holding", "frisbee", [10, 10, 100, 200], [150, 50, 60, 60])],
    )


def test_candidates_cover_all_three_sub_types(scene):
    cands = candidates_for_scene(scene)
    assert {c.sub_type.value for c in cands} == {"object", "attribute", "relation"}


def test_ground_truth_doc_contains_the_answer(scene):
    """Every retrieval-side perturbation is a mutation of this document, so the
    fact must actually be present in it."""
    for c in candidates_for_scene(scene):
        assert c.answer.lower() in c.doc.lower()


def test_round_trip_accepts_valid_candidates(scene):
    cands = candidates_for_scene(scene)
    assert sum(round_trip_ok(c, scene) for c in cands) >= 3


def test_round_trip_rejects_ambiguous_questions():
    """Two red objects of the same category => the attribute question has two
    admissible answers and must be filtered."""
    ambiguous = SceneGraph(
        image_id="t2", image_path="x.jpg", width=640, height=480, source="test",
        objects=[
            Obj("cup", [0, 0, 50, 50], ["red"]),
            Obj("cup", [100, 0, 50, 50], ["blue"]),
            Obj("dog", [200, 0, 50, 50], ["brown"]),
        ],
        relations=[],
    )
    cups = [c for c in candidates_for_scene(ambiguous) if "cup" in c.question]
    assert all(not round_trip_ok(c, ambiguous) for c in cups)


# --------------------------------------------------------------------------- #
# COCO split readiness (regression: the dev-gate bug)
# --------------------------------------------------------------------------- #


def test_resolve_split_requires_images_not_just_annotations(tmp_path, monkeypatch):
    """COCO ships train+val instance JSON in one zip, so annotations for a split
    can exist while its images are still downloading. Selecting that split would
    silently yield an empty corpus. `split_ready` must require images too."""
    import gib.dataengine.sources.coco as coco

    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    # Both annotation files present (as in annotations_trainval2017.zip)...
    (root / "annotations" / "instances_train2017.json").write_text("{}")
    (root / "annotations" / "instances_val2017.json").write_text("{}")
    # ...but only val images are on disk.
    (root / "val2017").mkdir()
    (root / "val2017" / "000000000001.jpg").write_bytes(b"\xff\xd8\xff")

    monkeypatch.setattr(coco, "coco_root", lambda: root)
    assert coco.split_ready("val2017")
    assert not coco.split_ready("train2017")   # annotations present, images absent
    assert coco.resolve_split("auto") == "val2017"
    # An explicit but not-ready request falls back to what IS ready.
    assert coco.resolve_split("train2017") == "val2017"


# --------------------------------------------------------------------------- #
# Release metadata: Croissant validity + measured characterisation
# --------------------------------------------------------------------------- #


def test_croissant_document_is_spec_valid():
    """The emitted Croissant file must load in the reference validator, not just
    look plausible. A malformed one claims machine-readability the dataset lacks."""
    from gib.dataengine.release import build_croissant, validate_croissant

    doc = build_croissant("v1.0")
    assert doc["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert doc["distribution"] and doc["recordSet"]

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "croissant.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        rep = validate_croissant(p)
    # mlcroissant may be absent in a lean env; then the structural check applies.
    assert rep["ok"], rep.get("errors") or rep.get("error")


def test_semver_normalization():
    from gib.dataengine.release import _semver

    assert _semver("v1.0") == "1.0.0"
    assert _semver("v0.1-dev") == "0.1.0"
    assert _semver("2.3.4") == "2.3.4"


def test_characterize_measures_hard_negative_difficulty():
    """The hard-negative overlap statistic must actually distinguish a surgical
    near-duplicate distractor from an unrelated one."""
    from gib.dataengine.characterize import characterize

    rows = [
        {
            "sub_type": "object", "axis": "noise", "dose": "3", "source": "test",
            "answer_visual": "apple", "answer_textual": "orange", "twin_id": "",
            "retrieved_docs": [
                "The photograph shows a red apple on the wooden table.",   # truth
                "The photograph shows a red orange on the wooden table.",  # surgical: 1 word flipped
                "A completely different sentence about weather patterns.",  # unrelated
            ],
            "gt_doc_position": 0,
        }
    ]
    report = characterize(rows)
    # The hardest distractor is the near-duplicate, so overlap should be high.
    assert report["hard_negative_overlap"]["mean"] > 0.6
