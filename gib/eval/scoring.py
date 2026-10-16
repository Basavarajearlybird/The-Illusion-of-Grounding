"""Answer scoring (Layer 2.3) -- a three-stage cascade.

Free-form VLM answers cannot be exact-matched: "a red one", "red.", "the umbrella
is red" and "red" are the same answer. The cascade escalates only as far as it
must, which matters at 60K x 6 models because the expensive stage is the last one:

1. **normalised exact match** -- lowercase, strip articles/punctuation, drop a
   leading restatement of the question. Resolves the large majority.
2. **alias match** -- WikiData aliases and WordNet synonyms ("couch"/"sofa",
   "grey"/"gray"). Cheap, and removes a whole class of spurious errors.
3. **LLM-as-judge** -- a pinned judge with a frozen rubric, consulted only for
   what survives.

Every answer is judged **twice**: once against ``answer_visual`` and once against
``answer_textual``. That double verdict is what the failure taxonomy consumes --
"wrong" and "wrong in the specific way the poisoned passage predicted" are
different findings, and only the second is Knowledge Inertia.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache

from gib.dataengine.sources.wikidata import default_graph

_ARTICLES = {"a", "an", "the"}
_FILLER = {
    "answer", "is", "are", "it", "its", "there", "this", "that", "of", "in",
    "on", "shows", "showing", "image", "picture", "photograph", "photo",
}
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, drop articles and answer-preamble filler."""
    t = (text or "").lower().strip()
    t = re.sub(r"^(the\s+)?answer\s*(is|:)\s*", "", t)
    t = t.translate(_PUNCT)
    tokens = [w for w in t.split() if w not in _ARTICLES]
    if len(tokens) > 1:
        stripped = [w for w in tokens if w not in _FILLER]
        if stripped:
            tokens = stripped
    return " ".join(tokens).strip()


@lru_cache(maxsize=8192)
def alias_set(term: str) -> frozenset[str]:
    """All surface forms that count as ``term``."""
    base = normalize(term)
    if not base:
        return frozenset()
    out = {base}
    out.update(normalize(a) for a in default_graph().aliases(base))
    try:
        from nltk.corpus import wordnet

        for syn in wordnet.synsets(base.replace(" ", "_")):
            for lemma in syn.lemmas():
                out.add(normalize(lemma.name().replace("_", " ")))
    except Exception:
        pass  # WordNet is optional; the curated alias table carries the load
    return frozenset(o for o in out if o)


def exact_match(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    return bool(g) and p == g


def alias_match(pred: str, gold: str) -> bool:
    p = normalize(pred)
    if not p:
        return False
    golds = alias_set(gold)
    if p in golds:
        return True
    # A short answer embedded in a sentence ("it is a red umbrella" vs "red").
    # The ceiling is applied to the RAW token count, not the normalised one:
    # normalisation strips filler words, so a long hedging answer can shrink
    # below the ceiling and get credited for merely containing the gold token.
    if len(pred.split()) <= 8:
        return any(re.search(rf"\b{re.escape(g)}\b", p) for g in golds if g)
    return False


@dataclass
class ScoreResult:
    matched: bool
    stage: str          # exact | alias | judge | none
    judge_raw: str = ""


def score_cascade(pred: str, gold: str, judge=None, *, question: str = "") -> ScoreResult:
    """Run the cascade against one gold answer."""
    if not gold:
        return ScoreResult(False, "none")
    if exact_match(pred, gold):
        return ScoreResult(True, "exact")
    if alias_match(pred, gold):
        return ScoreResult(True, "alias")
    if judge is not None:
        verdict, raw = judge(question=question, prediction=pred, gold=gold)
        return ScoreResult(verdict, "judge", raw)
    return ScoreResult(False, "none")


def is_refusal(pred: str) -> bool:
    """Refusals and non-answers.

    Kept distinct from wrong answers: a model that declines under heavy blur is
    behaving *correctly* about its own uncertainty, and folding that into
    Reasoning Collapse would misrepresent the taxonomy.
    """
    p = normalize(pred)
    if not p:
        return True
    patterns = (
        "cannot determine", "cant determine", "unable to", "not possible to tell",
        "cannot tell", "cant tell", "unclear", "i dont know", "unknown",
        "no information", "cannot be determined", "not enough information",
        "impossible to determine",
    )
    return any(pat in p for pat in patterns)
