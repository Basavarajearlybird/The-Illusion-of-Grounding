"""CHAIR -- Caption Hallucination Assessment with Image Relevance.

Rohrbach et al., "Object Hallucination in Image Captioning" (EMNLP 2018).

    CHAIR_i = (# hallucinated object mentions) / (# object mentions)
    CHAIR_s = (# captions containing >=1 hallucination) / (# captions)

**Where it applies in GIB-15K.** The visual-ambiguity axis, run in captioning
mode on degraded images. This is where CHAIR earns its place: it is the metric
that detects the transition from *seeing* to *guessing*, which is precisely what
the Resolution Threshold is defined as -- the sigma or crop% at which CHAIR jumps
more than 10 points over the clean condition.

Ground truth is the COCO instance inventory for the image. Mentions are extracted
with the standard synonym-mapped word matching from the original work, so
"bike" counts as "bicycle" and a caption is not penalised for vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from gib.dataengine.sources.coco import COCO_CATEGORIES

# Synonym map after Rohrbach et al.; without it, CHAIR massively over-reports
# hallucination on any model that does not happen to use COCO's exact wording.
SYNONYMS: dict[str, str] = {
    "bike": "bicycle", "motorbike": "motorcycle", "aeroplane": "airplane",
    "plane": "airplane", "jet": "airplane", "tv": "tv", "television": "tv",
    "monitor": "tv", "screen": "tv", "computer": "laptop", "notebook": "laptop",
    "cellphone": "cell phone", "mobile phone": "cell phone", "phone": "cell phone",
    "smartphone": "cell phone", "sofa": "couch", "settee": "couch",
    "table": "dining table", "desk": "dining table", "plant": "potted plant",
    "houseplant": "potted plant", "ball": "sports ball", "doughnut": "donut",
    "hotdog": "hot dog", "man": "person", "woman": "person", "men": "person",
    "women": "person", "boy": "person", "girl": "person", "child": "person",
    "children": "person", "people": "person", "guy": "person", "lady": "person",
    "kid": "person", "player": "person", "puppy": "dog", "kitten": "cat",
    "cellular telephone": "cell phone", "remote control": "remote",
    "teddy": "teddy bear", "bear": "bear", "cup": "cup", "mug": "cup",
    "glass": "wine glass", "bicycles": "bicycle",
}


def _plural_forms(word: str) -> set[str]:
    forms = {word}
    if word.endswith("s"):
        forms.add(word[:-1])
    else:
        forms.add(word + "s")
        if word.endswith(("ch", "sh", "x", "z")):
            forms.add(word + "es")
        if word.endswith("y") and word[-2:-1] not in "aeiou":
            forms.add(word[:-1] + "ies")
    return forms


def _build_lexicon() -> dict[str, str]:
    """surface form -> canonical COCO category."""
    lex: dict[str, str] = {}
    for cat in COCO_CATEGORIES:
        for form in _plural_forms(cat):
            lex[form] = cat
    for syn, cat in SYNONYMS.items():
        if cat in COCO_CATEGORIES:
            for form in _plural_forms(syn):
                lex.setdefault(form, cat)
    return lex


LEXICON = _build_lexicon()
# Longest-first so "sports ball" wins over "ball" and "dining table" over "table".
_SORTED_TERMS = sorted(LEXICON, key=len, reverse=True)
_MENTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SORTED_TERMS) + r")\b", flags=re.IGNORECASE
)


def extract_mentions(caption: str) -> list[str]:
    """Canonical COCO categories mentioned in ``caption`` (with repeats)."""
    out: list[str] = []
    for m in _MENTION_RE.finditer(caption.lower()):
        cat = LEXICON.get(m.group(1).lower())
        if cat:
            out.append(cat)
    return out


@dataclass
class ChairResult:
    chair_i: float
    chair_s: float
    n_captions: int
    n_mentions: int
    n_hallucinated: int
    hallucinated_objects: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chair_i": round(self.chair_i, 4),
            "chair_s": round(self.chair_s, 4),
            "n_captions": self.n_captions,
            "n_mentions": self.n_mentions,
            "n_hallucinated": self.n_hallucinated,
        }


def compute_chair(captions: list[str], gt_objects: list[set[str]]) -> ChairResult:
    """CHAIR_i and CHAIR_s over a set of (caption, ground-truth inventory) pairs."""
    if len(captions) != len(gt_objects):
        raise ValueError("captions and gt_objects must be the same length")

    total_mentions = 0
    total_halluc = 0
    captions_with_halluc = 0
    halluc_counter: dict[str, int] = {}

    for caption, gt in zip(captions, gt_objects):
        mentions = extract_mentions(caption)
        total_mentions += len(mentions)
        bad = [m for m in mentions if m not in gt]
        total_halluc += len(bad)
        if bad:
            captions_with_halluc += 1
            for b in bad:
                halluc_counter[b] = halluc_counter.get(b, 0) + 1

    n = len(captions)
    return ChairResult(
        chair_i=total_halluc / total_mentions if total_mentions else 0.0,
        chair_s=captions_with_halluc / n if n else 0.0,
        n_captions=n,
        n_mentions=total_mentions,
        n_hallucinated=total_halluc,
        hallucinated_objects=dict(sorted(halluc_counter.items(), key=lambda kv: -kv[1])),
    )


def resolution_threshold(
    chair_by_dose: dict[float, float], clean_chair: float, jump_points: float = 10.0
) -> float | None:
    """The Resolution Threshold: the first dose where CHAIR exceeds clean by
    more than ``jump_points`` percentage points.

    Returns ``None`` when the model never crosses the threshold across the tested
    range -- reported as ">16" (or ">40px", "<5%") in the paper rather than
    silently imputed, because "never degraded enough to break" is a real and
    reportable outcome.
    """
    for dose in sorted(chair_by_dose):
        if (chair_by_dose[dose] - clean_chair) * 100.0 > jump_points:
            return dose
    return None
