"""Persistent response cache (Layer 2.1) -- non-negotiable for the API tier.

Keyed on ``(model, variant_id, prompt_hash, temperature, sample_idx)``. Two
properties follow, and both are load-bearing for the project:

* no prompt is ever paid for twice, which is what makes a 60K-item benchmark
  financially survivable;
* a crashed run resumes for free, which matters because a 60K-item x 6-model
  campaign *will* crash at least once.

``prompt_hash`` is part of the key rather than just ``variant_id`` so that
editing the prompt template correctly invalidates the cache instead of silently
serving responses generated under the old framing.

SQLite in WAL mode: concurrent readers, one writer, no server, survives kill -9.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from gib.config import CACHE_DIR
from gib.schema import stable_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    variant_id  TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    temperature REAL NOT NULL,
    sample_idx  INTEGER NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_variant ON responses(model, variant_id);
"""


class ResponseCache:
    def __init__(self, path: str | Path | None = None):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = Path(path or CACHE_DIR / "responses.sqlite")
        self._local = threading.local()
        with self._conn() as con:
            con.executescript(_SCHEMA)
        self.hits = 0
        self.misses = 0

    def _conn(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            self._local.con = con
        return con

    @staticmethod
    def make_key(
        model: str, variant_id: str, prompt_hash: str, temperature: float, sample_idx: int
    ) -> str:
        return stable_hash(
            {
                "model": model,
                "variant_id": variant_id,
                "prompt_hash": prompt_hash,
                "temperature": round(float(temperature), 4),
                "sample_idx": int(sample_idx),
            }
        )

    def get(
        self, model: str, variant_id: str, prompt_hash: str, temperature: float, sample_idx: int = 0
    ) -> dict[str, Any] | None:
        key = self.make_key(model, variant_id, prompt_hash, temperature, sample_idx)
        row = self._conn().execute(
            "SELECT payload FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(
        self,
        model: str,
        variant_id: str,
        prompt_hash: str,
        temperature: float,
        sample_idx: int,
        payload: dict[str, Any],
    ) -> None:
        key = self.make_key(model, variant_id, prompt_hash, temperature, sample_idx)
        con = self._conn()
        with con:
            con.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?)",
                (
                    key, model, variant_id, prompt_hash, float(temperature),
                    int(sample_idx), json.dumps(payload), time.time(),
                ),
            )

    def count(self, model: str | None = None) -> int:
        con = self._conn()
        if model:
            return con.execute(
                "SELECT COUNT(*) FROM responses WHERE model = ?", (model,)
            ).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "rows": self.count(),
        }
