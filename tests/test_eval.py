"""Layer 2 tests: the scoring cascade, the taxonomy, the metrics, and the cache.

The taxonomy tests are the important ones. The classifier turns two judge
verdicts into a scientific claim about *how* a model failed, so a regression
there would silently change every taxonomy proportion in the paper.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gib.eval.metrics.calibration import expected_calibration_error, overconfidence_rate
from gib.eval.metrics.chair import compute_chair, extract_mentions, resolution_threshold
from gib.eval.metrics.pope import parse_yes_no, score_pope, sycophancy_slope
from gib.eval.metrics.refind import csr_from_logprobs
from gib.eval.metrics.robustness import fit_decay, robustness_score
from gib.eval.scoring import alias_match, exact_match, is_refusal, normalize
from gib.eval.taxonomy import TaxonomyInput, classify
from gib.eval.trust import self_consistency, trust_index_track_b
from gib.schema import FailureMode

# --------------------------------------------------------------------------- #
# Scoring cascade
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pred,gold",
    [
        ("red", "red"), ("Red.", "red"), ("The answer is red", "red"),
        ("a red", "red"), ("  RED  ", "red"),
    ],
)
def test_normalized_exact_match(pred, gold):
    assert exact_match(pred, gold)


def test_alias_match_handles_synonyms():
    assert alias_match("sofa", "couch")
    assert alias_match("grey", "gray")
    assert alias_match("bike", "bicycle")


def test_alias_match_rejects_category_siblings():
    """The core distinction the benchmark rests on: a different member of the
    same category is WRONG, not a synonym."""
    assert not alias_match("orange", "apple")
    assert not alias_match("green", "red")


def test_alias_match_does_not_credit_rambling_answers():
    long_answer = " ".join(["the", "image", "may", "contain", "many", "things",
                            "possibly", "including", "a", "red", "thing"])
    assert not alias_match(long_answer, "red")


def test_refusal_detection():
    assert is_refusal("I cannot determine this from the image")
    assert is_refusal("unclear")
    assert not is_refusal("red")


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #


def test_correct_when_matching_visual():
    r = classify(TaxonomyInput("conflict", True, False, "apple", "orange"))
    assert r is FailureMode.CORRECT


def test_knowledge_inertia_when_matching_injected_falsehood():
    r = classify(TaxonomyInput("conflict", False, True, "orange", "orange"))
    assert r is FailureMode.KNOWLEDGE_INERTIA


def test_knowledge_inertia_requires_an_actual_falsehood():
    """Without answer_textual the label would be unfalsifiable."""
    r = classify(TaxonomyInput("noise", False, True, "orange", ""))
    assert r is FailureMode.REASONING_COLLAPSE


def test_sensory_capture_only_on_the_visual_axis():
    capture = classify(
        TaxonomyInput("visual", False, False, "banana", "", degradation="gaussian_blur")
    )
    assert capture is FailureMode.SENSORY_CAPTURE
    # No degraded percept off the visual axis => the label is not available.
    assert classify(TaxonomyInput("noise", False, False, "banana", "")) is \
        FailureMode.REASONING_COLLAPSE


def test_refusal_under_blur_is_not_sensory_capture():
    """Declining to guess is the opposite failure from committing to a bad
    percept, and must not be counted as capture."""
    r = classify(
        TaxonomyInput("visual", False, False, "I cannot tell", "",
                      degradation="gaussian_blur")
    )
    assert r is FailureMode.REASONING_COLLAPSE


def test_reasoning_collapse_for_off_topic_output():
    r = classify(TaxonomyInput("conflict", False, False, "the weather is nice", "orange"))
    assert r is FailureMode.REASONING_COLLAPSE


# --------------------------------------------------------------------------- #
# Trust Index
# --------------------------------------------------------------------------- #


def test_track_b_trust_index_is_the_textual_share():
    samples = ["orange"] * 7 + ["apple"] * 3
    tr = trust_index_track_b(
        samples, answer_visual="apple", answer_textual="orange",
        matcher=lambda p, g: normalize(p) == normalize(g),
    )
    assert math.isclose(tr.trust_index, 0.7)
    assert tr.track == "B"


def test_track_b_ignores_off_manifold_samples():
    """Samples matching neither candidate would otherwise drag the index toward
    0.5 and mimic balance where there is none."""
    samples = ["orange"] * 6 + ["apple"] * 2 + ["I don't know"] * 2
    tr = trust_index_track_b(
        samples, answer_visual="apple", answer_textual="orange",
        matcher=lambda p, g: normalize(p) == normalize(g),
    )
    assert math.isclose(tr.trust_index, 0.75)


def test_self_consistency_normalizes_surface_forms():
    assert math.isclose(self_consistency(["red", "Red.", "red", "blue"]), 0.75)


# --------------------------------------------------------------------------- #
# CHAIR / POPE / REFIND
# --------------------------------------------------------------------------- #


def test_chair_counts_hallucinated_objects():
    captions = ["The image shows a dog and a bicycle.", "A cat sits on a couch."]
    gt = [{"dog"}, {"cat", "couch"}]
    res = compute_chair(captions, gt)
    assert res.n_hallucinated == 1          # 'bicycle'
    assert math.isclose(res.chair_s, 0.5)


def test_chair_synonym_map_prevents_false_hallucinations():
    """A model saying 'sofa' for 'couch' must not be scored as hallucinating."""
    res = compute_chair(["A sofa and a bike."], [{"couch", "bicycle"}])
    assert res.n_hallucinated == 0


def test_chair_prefers_longest_match():
    assert "sports ball" in extract_mentions("a sports ball on the field")


def test_resolution_threshold_finds_the_10_point_jump():
    assert resolution_threshold({1: 0.12, 2: 0.15, 4: 0.28, 8: 0.40}, clean_chair=0.10) == 4
    assert resolution_threshold({1: 0.11, 2: 0.12}, clean_chair=0.10) is None


@pytest.mark.parametrize(
    "text,expected",
    [("Yes", True), ("no.", False), ("There is a dog", True),
     ("There is no dog", False), ("maybe", None)],
)
def test_pope_parsing(text, expected):
    assert parse_yes_no(text) == expected


def test_pope_scores_and_yes_ratio():
    res = score_pope(["yes", "yes", "no", "no"], [True, False, True, False])
    assert math.isclose(res.accuracy, 0.5)
    assert math.isclose(res.yes_ratio, 0.5)


def test_pope_unparsed_counted_as_no():
    res = score_pope(["maybe"], [True])
    assert res.n_unparsed == 1
    assert res.accuracy == 0.0


def test_sycophancy_slope_detects_rising_yes_rate():
    assert sycophancy_slope({1: 0.5, 5: 0.6, 10: 0.7, 20: 0.9}) > 0


def test_csr_positive_when_context_lifts_the_token():
    mean, mx, per_token = csr_from_logprobs([-0.5, -0.4], [-2.0, -1.8])
    assert mean > 0 and len(per_token) == 2


# --------------------------------------------------------------------------- #
# Calibration + robustness
# --------------------------------------------------------------------------- #


def test_perfect_calibration_has_near_zero_ece():
    conf = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95] * 40)
    rng = np.random.default_rng(0)
    corr = (rng.random(conf.size) < conf).astype(float)
    assert expected_calibration_error(conf, corr).ece < 0.08


def test_confidently_wrong_produces_high_ece():
    conf = np.full(200, 0.95)
    corr = np.zeros(200)
    assert expected_calibration_error(conf, corr).ece > 0.9


def test_ece_drops_nan_rather_than_imputing():
    cal = expected_calibration_error([0.9, float("nan"), 0.8], [1.0, 1.0, 0.0])
    assert cal.n == 2


def test_overconfidence_rate():
    assert math.isclose(
        overconfidence_rate([0.95, 0.9, 0.2, 0.99], [0.0, 0.0, 0.0, 1.0], threshold=0.8),
        2 / 3,
    )


def test_robustness_penalizes_miscalibration():
    good = robustness_score(0.9, 0.6, ece=0.02)
    bad = robustness_score(0.9, 0.6, ece=0.40)
    assert good.r > bad.r
    assert math.isclose(good.ratio, bad.ratio)


def test_robustness_is_nan_without_a_confidence_signal():
    """A missing ECE must not silently become a perfect one."""
    assert math.isnan(robustness_score(0.9, 0.6, ece=float("nan")).r)


def test_decay_fit_recovers_lambda():
    doses = np.array([0, 1, 3, 5, 10, 20], dtype=float)
    true_lam = 0.12
    acc = 0.6 * np.exp(-true_lam * doses) + 0.25
    fit = fit_decay(doses, acc)
    assert fit.converged
    assert abs(fit.lam - true_lam) < 0.02
    assert fit.r_squared > 0.99


def test_decay_fit_reports_failure_on_too_few_points():
    fit = fit_decay([0, 1], [0.9, 0.8])
    assert not fit.converged
