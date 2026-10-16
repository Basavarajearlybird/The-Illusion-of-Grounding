"""Tests for the reproducibility gate: provenance, audit, and claim grading.

These guard the guarantee that a manuscript number cannot silently disagree with
what the pipeline measured — the single most important reproducibility property
of a benchmark paper.
"""

from __future__ import annotations

from gib.analysis.audit import audit_text, render
from gib.analysis.claims import grade
from gib.analysis.provenance import flatten, resolve

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_flatten_produces_dotted_keys():
    table = flatten({"fig1": {"model_a": {"lambda": 0.12}}, "n": 5})
    assert table["fig1.model_a.lambda"] == 0.12
    assert table["n"] == 5


def test_flatten_indexes_lists():
    table = flatten({"models": ["a", "b"]})
    assert table["models.0"] == "a" and table["models.1"] == "b"


def test_resolve_suggests_near_misses():
    import pytest

    table = {"fig1.model_a.lambda": 0.12}
    with pytest.raises(KeyError, match="prefix"):
        resolve("fig1.model_a.lamda", table)  # typo


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #


def test_render_fills_placeholders_from_provenance():
    table = {"decay.m.lambda": 0.1234, "n": 59648}
    text = "lambda = {{decay.m.lambda:.3f}} over {{n}} rows"
    out, errors = render(text, table)
    assert not errors
    assert out == "lambda = 0.123 over 59648 rows"


def test_render_reports_unresolved_keys():
    _out, errors = render("{{missing.key}}", {"present": 1})
    assert errors and "missing.key" in errors[0]


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def test_audit_passes_when_inline_value_matches():
    table = {"decay.m.lambda": 0.123}
    r = audit_text("the decay rate was 0.123 ⟨decay.m.lambda⟩", table)
    assert r.ok and r.n_inline_checked == 1


def test_audit_catches_stale_inline_value():
    """The whole point: a manuscript that says 0.87 when provenance says 0.12
    must fail the gate."""
    table = {"ads.value": 0.12}
    r = audit_text("ADS = 0.87 ⟨ads.value⟩", table)
    assert not r.ok and r.mismatches


def test_audit_catches_unresolved_placeholder():
    r = audit_text("value is {{nonexistent.key}}", {"other": 1})
    assert not r.ok and r.unresolved


def test_audit_tolerates_rounding():
    table = {"x": 0.12345}
    r = audit_text("0.1235 ⟨x⟩", table, rel_tol=1e-2)
    assert r.ok


# --------------------------------------------------------------------------- #
# Claim downgrade
# --------------------------------------------------------------------------- #


def test_strong_effect_gets_strong_verb():
    c = grade(p_value=1e-9, effect_size=0.95, n=80)
    assert c.tier == "strong"
    assert c.verb == "demonstrate"


def test_weak_effect_gets_hedged_verb():
    c = grade(p_value=0.01, effect_size=0.25, n=80)
    assert c.tier == "weak"
    assert c.verb == "weakly support"


def test_nonsignificant_becomes_null_claim():
    c = grade(p_value=0.3, effect_size=0.6, n=80)
    assert c.tier == "null"
    assert c.verb == "do not support"


def test_verbs_read_as_transitive_noun_phrase_takers():
    """Every verb must slot into 'we {verb} a collapse' without an adverb break."""
    for eff, tier in [(0.95, "strong"), (0.6, "moderate"), (0.25, "weak")]:
        c = grade(p_value=1e-4, effect_size=eff, n=80)
        assert c.tier == tier
        sentence = f"we {c.verb} a collapse in attention"
        assert " weakly weakly" not in sentence  # no double-hedge
        assert sentence.count("  ") == 0          # no adverb-gap artefact


def test_tiny_sample_is_not_over_claimed():
    c = grade(p_value=1e-6, effect_size=0.9, n=5)
    assert c.tier in ("marginal", "null")


def test_significant_but_negligible_is_marginal():
    """Large n can make a trivial effect 'significant'; the verb must stay weak."""
    c = grade(p_value=1e-4, effect_size=0.05, n=5000)
    assert c.tier == "marginal"
