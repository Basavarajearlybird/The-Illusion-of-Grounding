"""Claim-downgrade rule.

From the build directive: *if a real measurement is weaker than the draft,
downgrade language gracefully rather than sounding like a failure.* A benchmark
paper loses credibility both by over-claiming ("confirms", "proves") on a weak
effect and by hedging on a strong one. This module maps a measured effect onto
calibrated language so the manuscript's verbs track the evidence automatically.

The calibration is deliberately conservative and is driven by two standard
quantities — a significance level and an effect size — not by whether the result
happened to match the draft. It is used at render time: a template writes
``models {{claim.vam_collapse.verb}} a collapse in visual attention`` and the verb
is filled from the actual statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

# Strength tiers, strongest first. Verbs are transitive and take a NOUN PHRASE
# object ("we {verb} a collapse in attention"; "the evidence {verb_singular} the
# hypothesis that ..."), so a sentence reads cleanly at any strength without an
# awkwardly-placed adverb. (name, plural verb, singular verb)
_TIERS = [
    ("strong",   "demonstrate",         "demonstrates"),
    ("moderate", "support",             "supports"),
    ("weak",     "weakly support",      "weakly supports"),
    ("marginal", "are consistent with", "is consistent with"),
    ("null",     "do not support",      "does not support"),
]
_VERB = {name: verb for name, verb, _ in _TIERS}
_SINGULAR = {name: sing for name, _, sing in _TIERS}


@dataclass
class Claim:
    tier: str
    p_value: float
    effect_size: float
    n: int

    @property
    def verb(self) -> str:
        """Plural transitive verb: 'we {verb} <noun phrase>'."""
        return _VERB[self.tier]

    @property
    def verb_singular(self) -> str:
        """Singular transitive verb: 'the evidence {verb_singular} <noun phrase>'."""
        return _SINGULAR[self.tier]

    # Retained for backward compatibility; now identical to `verb` since the verbs
    # are self-hedging (weakly support / are consistent with / do not support).
    @property
    def hedged_verb(self) -> str:
        return _VERB[self.tier]


def grade(
    *,
    p_value: float,
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    small: float = 0.2,
    medium: float = 0.5,
    large: float = 0.8,
    min_n: int = 20,
) -> Claim:
    """Map (p, |effect|, n) onto a strength tier.

    Thresholds follow Cohen's conventions for the effect size and a standard
    alpha for significance. The effect size is taken as an absolute magnitude, so
    the same grading serves a collapse (negative) and an increase (positive).
    """
    e = abs(effect_size)

    if n < min_n or p_value != p_value:  # too small, or NaN p
        tier = "marginal" if e >= small else "null"
    elif p_value > alpha:
        tier = "null"
    elif e >= large:
        tier = "strong"
    elif e >= medium:
        tier = "moderate"
    elif e >= small:
        tier = "weak"
    else:
        # Significant but negligible effect — significance from large n alone.
        tier = "marginal"

    return Claim(tier=tier, p_value=p_value, effect_size=effect_size, n=n)


def claim_fields(prefix: str, claim: Claim) -> dict[str, object]:
    """Expose a graded claim as flat provenance keys for the audit/render step,
    e.g. ``claim.vam_collapse.verb`` -> "demonstrate"."""
    return {
        f"{prefix}.tier": claim.tier,
        f"{prefix}.verb": claim.verb,
        f"{prefix}.verb_singular": claim.verb_singular,
        f"{prefix}.hedged_verb": claim.hedged_verb,
        f"{prefix}.p_value": claim.p_value,
        f"{prefix}.effect_size": claim.effect_size,
        f"{prefix}.n": claim.n,
    }
