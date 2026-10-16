"""LLM-as-judge, with a frozen rubric and a circularity guard.

The architecture names judge circularity as one of two silent failure modes to
guard against from the start: never use a model from the zoo as the judge for its
own outputs without reporting the human-agreement check. That guard is enforced
here in code (:func:`assert_no_circularity`) rather than left to discipline,
because the failure is silent by nature -- a model grading itself produces
plausible numbers and no error.

Judge verdicts are cached on ``(judge_model, rubric_version, question, gold,
prediction)``. Duplicate answers are extremely common at 60K items ("red" recurs
constantly), so the cache typically removes the large majority of judge calls.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from gib.config import CACHE_DIR, CONFIG_DIR
from gib.schema import stable_hash

RUBRIC_VERSION = "rubric_v1"
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"


@lru_cache(maxsize=4)
def load_rubric(version: str = RUBRIC_VERSION) -> str:
    text = (Path(CONFIG_DIR) / "judge" / f"{version}.txt").read_text(encoding="utf-8")
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")).strip()


class JudgeCircularityError(RuntimeError):
    pass


def assert_no_circularity(judge_model: str, zoo_models: list[str]) -> None:
    """Refuse to let a model grade its own outputs.

    Matching is on model *family* prefix, not exact id: ``gpt-4o`` judging
    ``gpt-4o-mini`` is the same conflict of interest, and an exact-match check
    would wave it through.
    """
    def family(m: str) -> str:
        return re.split(r"[-_]", m.lower())[0]

    jf = family(judge_model)
    clashes = [m for m in zoo_models if family(m) == jf and not m.startswith("sim-")]
    if clashes:
        raise JudgeCircularityError(
            f"judge {judge_model!r} shares a model family with zoo members {clashes}. "
            f"Pick an out-of-zoo judge, or pass allow_circular=True and report the "
            f"judge-human agreement for those models explicitly."
        )


class JudgeCache:
    def __init__(self, path: Path | None = None):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or CACHE_DIR / "judge.sqlite"
        self._local = threading.local()
        with self._conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS verdicts "
                "(key TEXT PRIMARY KEY, verdict INTEGER, raw TEXT)"
            )

    def _conn(self):
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            self._local.con = con
        return con

    def get(self, key: str):
        row = self._conn().execute(
            "SELECT verdict, raw FROM verdicts WHERE key = ?", (key,)
        ).fetchone()
        return (bool(row[0]), row[1]) if row else None

    def put(self, key: str, verdict: bool, raw: str) -> None:
        con = self._conn()
        with con:
            con.execute(
                "INSERT OR REPLACE INTO verdicts VALUES (?,?,?)", (key, int(verdict), raw)
            )


class Judge:
    """Callable ``(question, prediction, gold) -> (bool, raw)``."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        rubric_version: str = RUBRIC_VERSION,
        *,
        backend: Callable[[str], str] | None = None,
        cache: JudgeCache | None = None,
        offline_ok: bool = False,
    ):
        self.model = model
        self.rubric_version = rubric_version
        self.rubric = load_rubric(rubric_version)
        self.cache = cache if cache is not None else JudgeCache()
        self.offline_ok = offline_ok
        self._backend = backend
        self.calls = 0
        self.cache_hits = 0
        # Once the backend is known to be unreachable, stop retrying it. Without
        # this latch an offline run pays a failed import or a network timeout on
        # every one of 60K items, which dominates the wall clock.
        self._unavailable = ""

    @property
    def backend(self) -> Callable[[str], str]:
        if self._backend is None:
            from gib.adapters.registry import get_text_model

            self._backend = get_text_model(self.model)
        return self._backend

    def __call__(self, *, question: str, prediction: str, gold: str) -> tuple[bool, str]:
        key = stable_hash(
            {
                "judge": self.model,
                "rubric": self.rubric_version,
                "q": question,
                "gold": gold,
                "pred": prediction,
            }
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit

        if self._unavailable:
            return False, self._unavailable

        prompt = self.rubric.format(question=question, gold=gold, prediction=prediction)
        try:
            raw = self.backend(prompt)
            self.calls += 1
        except Exception as exc:  # noqa: BLE001
            if self.offline_ok:
                # No judge available: fall through as NOMATCH and let the runner
                # record judge_verdict="unavailable", rather than crashing a
                # multi-hour campaign or silently crediting the answer.
                self._unavailable = f"judge_unavailable: {type(exc).__name__}"
                self.unavailable_hits = getattr(self, "unavailable_hits", 0) + 1
                return False, self._unavailable
            raise

        # NOMATCH is checked first: it contains "MATCH" as a substring, so a
        # naive `"MATCH" in raw` would score every rejection as a match.
        upper = raw.upper()
        verdict = bool(re.search(r"\bMATCH\b", upper)) and not re.search(r"NOMATCH", upper)
        self.cache.put(key, verdict, raw.strip()[:500])
        return verdict, raw.strip()[:500]

    def stats(self) -> dict:
        total = self.calls + self.cache_hits
        return {
            "judge_model": self.model,
            "rubric_version": self.rubric_version,
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / total, 4) if total else 0.0,
            # Non-zero means some items were scored by the cascade alone. Any
            # run reporting this must say so -- it is a weaker scorer.
            "unavailable": self._unavailable,
            "unavailable_hits": getattr(self, "unavailable_hits", 0),
        }


def judge_human_agreement(
    judge: Judge, annotated: list[dict]
) -> dict[str, float]:
    """Judge-vs-human agreement on the 200-item audit sample.

    Reported in the paper alongside every judge-scored number. ``annotated`` rows
    need ``question``, ``prediction``, ``gold`` and a human ``label`` (bool).
    Cohen's kappa is reported next to raw agreement because these labels are
    heavily skewed toward MATCH, and raw agreement alone would look impressive
    for a judge that simply always says MATCH.
    """
    if not annotated:
        return {}
    agree = 0
    a1 = a0 = b1 = b0 = 0
    for row in annotated:
        pred, _ = judge(
            question=row.get("question", ""),
            prediction=row["prediction"],
            gold=row["gold"],
        )
        human = bool(row["label"])
        agree += int(pred == human)
        a1 += int(pred); a0 += int(not pred)
        b1 += int(human); b0 += int(not human)

    n = len(annotated)
    po = agree / n
    pe = (a1 / n) * (b1 / n) + (a0 / n) * (b0 / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(po, 4), "cohens_kappa": round(kappa, 4)}
