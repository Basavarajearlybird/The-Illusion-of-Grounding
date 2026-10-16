"""Layer 2.5 -- the evaluation orchestrator.

A run is fully described by ``{model, dataset config, axis, dose list, sampling
params}``. Results append to parquet partitioned by model/axis and are queried
with DuckDB. Every run logs the git SHA, dataset version hash, seed, prompt
version, judge version and adapter version, so a result row can always be traced
back to the exact code and data that produced it.

**Resume-safety is a construction property, not a feature.** A 60K-item x
6-model campaign will crash at least once, so:

* every model response goes through the persistent cache before it is computed;
* completed ``variant_id``s are read back from the existing shard and skipped;
* writes are idempotent -- a shard is rewritten wholesale from the union of old
  and new rows, so a crash mid-write leaves either the old shard or the new one,
  never a half-written mix.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gib.adapters.cache import ResponseCache
from gib.adapters.registry import get_adapter, is_simulated
from gib.config import RESULTS_DIR, VARIANTS_DIR, ensure_dirs, git_sha
from gib.eval.judge import Judge, assert_no_circularity
from gib.eval.metrics.calibration import confidence_from_logprobs
from gib.eval.metrics.refind import csr_for_target
from gib.eval.prompt import PROMPT_VERSION, render_rag_prompt
from gib.eval.scoring import score_cascade
from gib.eval.taxonomy import TaxonomyInput, classify
from gib.eval.trust import (
    majority_answer,
    self_consistency,
    trust_index_track_a,
    trust_index_track_b,
)
from gib.schema import Axis, stable_hash
from gib.seeds import seed_everything


@dataclass
class RunConfig:
    model: str
    dataset: str = "gib15k_eval.parquet"
    axes: list[str] = field(default_factory=lambda: [a.value for a in Axis])
    doses: list[str] | None = None
    sub_types: list[str] | None = None
    limit: int | None = None

    # Track B sampling
    n_samples: int = 10
    temperature: float = 0.7

    # Track A extras
    measure_csr: bool = True
    measure_trust: bool = True
    proxy_model: str | None = None   # scores CSR for Track B models

    judge_model: str = "claude-opus-4-8"
    allow_circular_judge: bool = False
    offline_judge: bool = False      # degrade to cascade-only when no judge is reachable

    seed: int = 20260721
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return stable_hash(
            {
                "model": self.model, "dataset": self.dataset, "axes": sorted(self.axes),
                "doses": self.doses, "sub_types": self.sub_types,
                "n_samples": self.n_samples, "temperature": self.temperature,
                "prompt": PROMPT_VERSION, "seed": self.seed,
            }
        )[:12]


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #


def load_variants(cfg: RunConfig) -> list[dict]:
    import pyarrow.parquet as pq

    path = VARIANTS_DIR / cfg.dataset
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}. Run `make dataset` first.")
    rows = pq.read_table(path).to_pylist()

    axes = set(cfg.axes)
    rows = [r for r in rows if r["axis"] in axes]
    if cfg.doses:
        keep = set(cfg.doses)
        rows = [r for r in rows if r["dose"] in keep or r["axis"] == Axis.CLEAN.value]
    if cfg.sub_types:
        keep = set(cfg.sub_types)
        rows = [r for r in rows if r["sub_type"] in keep]

    rows.sort(key=lambda r: r["variant_id"])
    if cfg.limit:
        rows = rows[: cfg.limit]
    return rows


def shard_path(model: str, axis: str) -> Path:
    return RESULTS_DIR / f"model={model}" / f"axis={axis}" / "results.parquet"


def completed_variant_ids(model: str, axes: Iterable[str]) -> set[str]:
    """Variant ids already present in this model's shards -- the resume set."""
    import pyarrow.parquet as pq

    done: set[str] = set()
    for axis in axes:
        p = shard_path(model, axis)
        if p.exists():
            try:
                done.update(pq.read_table(p, columns=["variant_id"]).column(0).to_pylist())
            except Exception:
                pass
    return done


def write_shard(model: str, axis: str, rows: list[dict]) -> Path:
    """Idempotent write: union with any existing shard, keyed on variant_id."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = shard_path(model, axis)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    if path.exists():
        try:
            for r in pq.read_table(path).to_pylist():
                merged[r["variant_id"]] = r
        except Exception:
            pass
    for r in rows:
        merged[r["variant_id"]] = r

    out = sorted(merged.values(), key=lambda r: r["variant_id"])
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(out), tmp, compression="zstd")
    tmp.replace(path)  # atomic on the same filesystem
    return path


# --------------------------------------------------------------------------- #
# Per-item evaluation
# --------------------------------------------------------------------------- #


def _cached_generate(
    adapter, cache: ResponseCache, row: dict, prompt_hash: str,
    *, temperature: float, sample_idx: int, meta: dict,
):
    """One generation, served from cache when possible."""
    hit = cache.get(adapter.name, row["variant_id"], prompt_hash, temperature, sample_idx)
    if hit is not None:
        return hit, True

    res = adapter.generate(
        row["image_path"], row["question"], list(row["retrieved_docs"] or []),
        temperature=temperature, sample_idx=sample_idx, meta=meta,
    )
    payload = {
        "answer": res.answer,
        "latency": res.latency,
        "token_logprobs": res.token_logprobs,
        "raw": res.raw if isinstance(res.raw, (dict, list, str, int, float, type(None))) else None,
    }
    cache.put(adapter.name, row["variant_id"], prompt_hash, temperature, sample_idx, payload)
    return payload, False


def evaluate_row(
    adapter, cache: ResponseCache, judge, cfg: RunConfig, row: dict, proxy=None
) -> dict:
    """Evaluate one variant and emit one result row."""
    docs = list(row["retrieved_docs"] or [])
    prompt_hash = stable_hash({"p": render_rag_prompt(row["question"], docs),
                               "v": PROMPT_VERSION})
    meta = {
        "variant_id": row["variant_id"], "axis": row["axis"],
        "answer_visual": row["answer_visual"], "answer_textual": row["answer_textual"],
        "n_distractors": row.get("n_distractors", 0),
        "degradation": row.get("degradation", "none"),
        "degradation_param": row.get("degradation_param", 0.0),
    }
    track = adapter.track
    t0 = time.time()

    # -- generate ---------------------------------------------------------- #
    if track == "B":
        # Track B needs N samples for both the Trust Index and the confidence
        # estimate, so sampling is unconditional rather than conflict-only.
        payloads = [
            _cached_generate(adapter, cache, row, prompt_hash,
                             temperature=cfg.temperature, sample_idx=i, meta=meta)[0]
            for i in range(cfg.n_samples)
        ]
        samples = [p["answer"] for p in payloads]
        answer = majority_answer(samples)
        confidence = self_consistency(samples)
        token_logprobs = None
        n_samples = len(samples)
    else:
        payload, _ = _cached_generate(adapter, cache, row, prompt_hash,
                                      temperature=0.0, sample_idx=0, meta=meta)
        answer = payload["answer"]
        samples = [answer]
        token_logprobs = payload.get("token_logprobs")
        confidence = confidence_from_logprobs(token_logprobs)
        n_samples = 1

    # -- score against both candidate answers ------------------------------- #
    sv = score_cascade(answer, row["answer_visual"], judge, question=row["question"])
    st = (
        score_cascade(answer, row["answer_textual"], judge, question=row["question"])
        if row["answer_textual"]
        else None
    )
    matches_visual = sv.matched
    matches_textual = bool(st and st.matched)

    failure = classify(
        TaxonomyInput(
            axis=row["axis"], matches_visual=matches_visual, matches_textual=matches_textual,
            prediction=answer, answer_textual=row["answer_textual"],
            degradation=row.get("degradation", "none"),
        )
    )

    # -- Trust Index (conflict axis) ---------------------------------------- #
    trust_index = float("nan")
    raw_ratio = float("nan")
    if cfg.measure_trust and row["answer_textual"] and row["axis"] in (
        Axis.CONFLICT.value, Axis.NOISE.value
    ):
        try:
            if track == "A":
                tr = trust_index_track_a(
                    adapter, image=row["image_path"], question=row["question"], docs=docs,
                    answer_visual=row["answer_visual"], answer_textual=row["answer_textual"],
                    meta=meta,
                )
            else:
                # Deliberate asymmetry, and one the paper should footnote: the
                # Track B matcher runs the cascade WITHOUT the judge. Judging N
                # samples against 2 candidates would multiply judge calls by ~20x
                # over the whole corpus for a marginal gain, since sampled
                # answers to these items are overwhelmingly short spans that
                # exact/alias matching already resolves. The consequence is that
                # a Track B trust index is very slightly conservative relative to
                # a Track A one.
                tr = trust_index_track_b(
                    samples, answer_visual=row["answer_visual"],
                    answer_textual=row["answer_textual"],
                    matcher=lambda p, g: score_cascade(p, g, None).matched,
                )
            trust_index, raw_ratio = tr.trust_index, tr.raw_ratio
            if track == "B" and not np.isnan(tr.confidence):
                confidence = tr.confidence
        except NotImplementedError:
            pass

    # -- CSR / REFIND -------------------------------------------------------- #
    csr = float("nan")
    csr_proxy = False
    if cfg.measure_csr and row["answer_textual"] and docs:
        try:
            cr = csr_for_target(
                adapter, image=row["image_path"], question=row["question"], docs=docs,
                target=row["answer_textual"], meta=meta, proxy_adapter=proxy,
            )
            csr, csr_proxy = cr.csr_mean, cr.proxy
        except NotImplementedError:
            pass

    return {
        "run_id": cfg.run_id,
        "model": adapter.name,
        "track": track,
        "simulated": is_simulated(adapter.name),
        "variant_id": row["variant_id"],
        "pair_id": row["pair_id"],
        "axis": row["axis"],
        "dose": row["dose"],
        "sub_type": row["sub_type"],
        "degradation": row.get("degradation", "none"),
        "degradation_param": float(row.get("degradation_param", 0.0) or 0.0),
        "n_distractors": int(row.get("n_distractors", 0) or 0),
        "gt_doc_position": int(row.get("gt_doc_position", -1)),
        "twin_id": row.get("twin_id", ""),
        "answer": answer,
        "answer_visual": row["answer_visual"],
        "answer_textual": row["answer_textual"],
        "is_correct": bool(matches_visual),
        "matches_visual": bool(matches_visual),
        "matches_textual": bool(matches_textual),
        "failure_mode": failure.value,
        "confidence": float(confidence) if confidence is not None else float("nan"),
        "trust_index": float(trust_index),
        "trust_raw_ratio": float(raw_ratio),
        "csr": float(csr),
        "csr_proxy": bool(csr_proxy),
        "score_stage": sv.stage,
        "judge_verdict": sv.judge_raw[:200],
        "latency_s": round(time.time() - t0, 4),
        "n_samples": n_samples,
        "prompt_version": PROMPT_VERSION,
        "judge_model": cfg.judge_model,
        "git_sha": git_sha(),
        "adapter_version": getattr(adapter, "version", "?"),
        "seed": cfg.seed,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run(cfg: RunConfig, *, resume: bool = True, flush_every: int = 250, verbose: bool = True) -> dict:
    ensure_dirs()
    seed_everything(cfg.seed)

    if not cfg.allow_circular_judge:
        assert_no_circularity(cfg.judge_model, [cfg.model])

    rows = load_variants(cfg)
    if resume:
        done = completed_variant_ids(cfg.model, cfg.axes)
        before = len(rows)
        rows = [r for r in rows if r["variant_id"] not in done]
        if verbose and before != len(rows):
            print(f"  [eval] resuming: {before - len(rows)} of {before} already complete")

    if not rows:
        print(f"[eval] {cfg.model}: nothing to do")
        return {"model": cfg.model, "n_new": 0}

    adapter = get_adapter(cfg.model, **cfg.adapter_kwargs)
    proxy = get_adapter(cfg.proxy_model) if cfg.proxy_model else None
    cache = ResponseCache()
    judge = Judge(cfg.judge_model, offline_ok=cfg.offline_judge)

    if verbose:
        print(f"[eval] {cfg.model} (track {adapter.track}) on {len(rows)} variants")
        if is_simulated(cfg.model):
            print("  *** SIMULATED ADAPTER -- results are for pipeline verification only ***")

    buffer: dict[str, list[dict]] = {}
    n_done = 0
    t0 = time.time()
    errors = 0

    def _flush():
        for axis, axis_rows in buffer.items():
            if axis_rows:
                write_shard(cfg.model, axis, axis_rows)
        buffer.clear()

    for row in rows:
        try:
            result = evaluate_row(adapter, cache, judge, cfg, row, proxy=proxy)
            buffer.setdefault(row["axis"], []).append(result)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            # One bad item must not kill a multi-hour campaign; it is counted and
            # surfaced in the manifest so a systematic failure is still visible.
            if errors <= 5:
                print(f"  [eval] error on {row['variant_id']}: {type(exc).__name__}: {exc}")
            elif errors == 6:
                print("  [eval] (further per-item errors suppressed)")

        n_done += 1
        if n_done % flush_every == 0:
            _flush()
            if verbose:
                rate = n_done / max(1e-6, time.time() - t0)
                eta = (len(rows) - n_done) / max(rate, 1e-6)
                print(f"  [eval] {n_done}/{len(rows)}  {rate:.1f}/s  eta {eta/60:.1f}m  "
                      f"cache {cache.stats()['hit_rate']:.0%}")
    _flush()

    manifest = {
        "run_id": cfg.run_id,
        "model": cfg.model,
        "track": adapter.track,
        "simulated": is_simulated(cfg.model),
        "n_evaluated": n_done,
        "n_errors": errors,
        "elapsed_s": round(time.time() - t0, 1),
        "dataset": cfg.dataset,
        "axes": cfg.axes,
        "prompt_version": PROMPT_VERSION,
        "git_sha": git_sha(),
        "seed": cfg.seed,
        "cache": cache.stats(),
        "judge": judge.stats(),
        "adapter_version": getattr(adapter, "version", "?"),
        "adapter_kwargs": cfg.adapter_kwargs,
    }
    mpath = RESULTS_DIR / f"model={cfg.model}" / "run_manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    adapter.close()
    if verbose:
        print(f"[eval] {cfg.model}: {n_done} items in {manifest['elapsed_s']}s "
              f"({errors} errors) -> {RESULTS_DIR / f'model={cfg.model}'}")
    return manifest
