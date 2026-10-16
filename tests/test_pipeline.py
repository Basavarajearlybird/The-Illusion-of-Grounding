"""Integration tests: the contracts *between* layers.

Unit tests catch broken functions; these catch the failures that only appear
where two layers meet — a cache key that ignores the prompt, a judge that grades
its own outputs, an eval that silently re-runs completed work.
"""

from __future__ import annotations

import numpy as np
import pytest

from gib.adapters.cache import ResponseCache
from gib.adapters.registry import available_models, get_adapter, is_simulated
from gib.eval.judge import JudgeCircularityError, assert_no_circularity
from gib.eval.prompt import render_context_block, render_rag_prompt
from gib.human.score import krippendorff_alpha
from gib.interp.hooks import vam_from_generation
from gib.interp.stats import benjamini_hochberg, cliffs_delta, paired_vam_test
from gib.schema import Axis, make_variant_id, stable_hash

# --------------------------------------------------------------------------- #
# Schema contract
# --------------------------------------------------------------------------- #


def test_variant_id_is_stable_and_unique_per_dose():
    a = make_variant_id("p1", Axis.NOISE, 5)
    assert a == make_variant_id("p1", Axis.NOISE, 5)
    assert a != make_variant_id("p1", Axis.NOISE, 10)
    assert a != make_variant_id("p1", Axis.VISUAL, 5)


def test_variant_id_survives_float_doses():
    assert "0p5" in make_variant_id("p1", Axis.VISUAL, 0.5)


def test_stable_hash_is_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def test_prompt_numbers_passages_so_position_is_visible():
    """gt_doc_position is only a meaningful covariate if the model can perceive
    position at all."""
    block = render_context_block(["first doc", "second doc"])
    assert "[1] first doc" in block and "[2] second doc" in block


def test_prompt_is_stable_across_calls():
    a = render_rag_prompt("What color?", ["doc"])
    b = render_rag_prompt("What color?", ["doc"])
    assert a == b and "What color?" in a


def test_prompt_handles_no_documents():
    assert "no passages" in render_rag_prompt("Q?", []).lower()


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "c.sqlite")


def test_cache_roundtrip(cache):
    assert cache.get("m", "v1", "ph", 0.0, 0) is None
    cache.put("m", "v1", "ph", 0.0, 0, {"answer": "red"})
    assert cache.get("m", "v1", "ph", 0.0, 0)["answer"] == "red"


def test_cache_key_includes_prompt_hash(cache):
    """Editing the prompt template must invalidate the cache, not silently serve
    responses generated under the old framing."""
    cache.put("m", "v1", "prompt_A", 0.0, 0, {"answer": "red"})
    assert cache.get("m", "v1", "prompt_B", 0.0, 0) is None


def test_cache_separates_samples_and_temperatures(cache):
    cache.put("m", "v1", "ph", 0.7, 0, {"answer": "a"})
    cache.put("m", "v1", "ph", 0.7, 1, {"answer": "b"})
    assert cache.get("m", "v1", "ph", 0.7, 0)["answer"] == "a"
    assert cache.get("m", "v1", "ph", 0.7, 1)["answer"] == "b"
    assert cache.get("m", "v1", "ph", 0.0, 0) is None


# --------------------------------------------------------------------------- #
# Judge circularity guard
# --------------------------------------------------------------------------- #


def test_circularity_guard_blocks_self_judging():
    with pytest.raises(JudgeCircularityError):
        assert_no_circularity("gpt-4o", ["gpt-4o", "llava-1.6-7b"])


def test_circularity_guard_matches_family_not_exact_id():
    """gpt-4o judging gpt-4o-mini is the same conflict of interest."""
    with pytest.raises(JudgeCircularityError):
        assert_no_circularity("gpt-4o", ["gpt-4o-mini"])


def test_circularity_guard_allows_out_of_zoo_judge():
    assert_no_circularity("claude-opus-4-8", ["gpt-4o", "gemini-1.5-pro"])


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def test_registry_exposes_the_full_zoo():
    models = available_models()
    for expected in ("llava-1.6-7b", "llava-1.6-13b", "qwen2.5-vl-7b",
                     "gpt-4o", "gemini-1.5-pro"):
        assert expected in models


def test_simulated_adapters_are_clearly_named():
    assert all(is_simulated(m) for m in available_models() if m.startswith("sim-"))
    assert not is_simulated("llava-1.6-7b")


def test_simulated_adapter_responds_to_dosage():
    """The fixture must actually respond to the stressor, or it exercises the
    analysis code against a constant."""
    a = get_adapter("sim-open-7b")
    low = a._p_correct({"axis": "noise", "n_distractors": 1})
    high = a._p_correct({"axis": "noise", "n_distractors": 20})
    assert low > high


def test_simulated_adapter_responds_to_blur():
    a = get_adapter("sim-open-7b")
    mild = a._p_correct({"axis": "visual", "degradation": "gaussian_blur",
                         "degradation_param": 1})
    severe = a._p_correct({"axis": "visual", "degradation": "gaussian_blur",
                           "degradation_param": 16})
    assert mild > severe


def test_simulated_adapter_is_deterministic():
    meta = {"variant_id": "v1", "axis": "conflict",
            "answer_visual": "apple", "answer_textual": "orange"}
    a = get_adapter("sim-open-7b")
    r1 = a.generate("", "q", [], meta=meta).answer
    r2 = get_adapter("sim-open-7b").generate("", "q", [], meta=meta).answer
    assert r1 == r2


def test_track_assignment_follows_capabilities():
    assert get_adapter("sim-open-7b").track == "A"      # logprobs
    assert get_adapter("sim-frontier-a").track == "B"   # sampling only


# --------------------------------------------------------------------------- #
# Interpretability
# --------------------------------------------------------------------------- #


def test_vam_reduction_shapes():
    steps, layers, heads = 5, 8, 4
    cap = {
        "vam_per_step_layer_head": np.random.rand(steps, layers, heads).astype(np.float32),
        "visual_span": np.array([10, 586]),
    }
    v = vam_from_generation(cap, "v1", "m")
    assert v.vam_per_layer.shape == (layers,)
    assert v.vam_per_layer_head.shape == (layers, heads)
    assert v.n_visual_tokens == 576


def test_vam_returns_none_on_empty_capture():
    assert vam_from_generation({}, "v1", "m") is None


def test_paired_test_detects_a_real_collapse():
    rng = np.random.default_rng(0)
    clean = {str(i): float(v) for i, v in enumerate(rng.normal(0.30, 0.03, 80))}
    inertia = {str(i): float(v) for i, v in enumerate(rng.normal(0.12, 0.03, 80))}
    t = paired_vam_test(clean, inertia)
    assert t.p_value < 0.001
    assert t.median_diff > 0.1
    assert t.effect_size_r > 0.8


def test_paired_test_reports_null_when_there_is_no_effect():
    rng = np.random.default_rng(1)
    a = {str(i): float(v) for i, v in enumerate(rng.normal(0.2, 0.05, 100))}
    b = {str(i): float(v) for i, v in enumerate(rng.normal(0.2, 0.05, 100))}
    assert paired_vam_test(a, b).p_value > 0.05


def test_cliffs_delta_bounds():
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)


def test_fdr_correction_rejects_isolated_marginal_hits():
    """One layer at p=0.04 among 99 nulls is what an uncorrected per-layer test
    would report as a finding; BH correctly declines it."""
    pvals = [0.04] + [0.9] * 99
    corrected = benjamini_hochberg(pvals, alpha=0.05)
    assert not any(corrected)
    assert sum(p < 0.05 for p in pvals) == 1


def test_fdr_keeps_a_strong_consistent_signal():
    pvals = [0.0001] * 20 + [0.9] * 20
    assert sum(benjamini_hochberg(pvals, alpha=0.05)) == 20


# --------------------------------------------------------------------------- #
# Human study
# --------------------------------------------------------------------------- #


def test_krippendorff_perfect_agreement():
    ratings = {"i1": {"p1": "red", "p2": "red"}, "i2": {"p1": "blue", "p2": "blue"}}
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_krippendorff_handles_missing_data():
    """Not every participant sees every item — the reason Krippendorff is used
    instead of Fleiss."""
    ratings = {
        "i1": {"p1": "red", "p2": "red"},
        "i2": {"p2": "blue", "p3": "blue"},
        "i3": {"p1": "green", "p3": "green"},
    }
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_krippendorff_disagreement_is_low():
    ratings = {
        "i1": {"p1": "red", "p2": "blue"},
        "i2": {"p1": "blue", "p2": "red"},
        "i3": {"p1": "red", "p2": "blue"},
    }
    assert krippendorff_alpha(ratings) < 0.1
