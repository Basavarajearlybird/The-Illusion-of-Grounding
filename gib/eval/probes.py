"""CHAIR and POPE evaluation passes.

These two metrics need a different *interaction mode* from the main QA loop --
CHAIR needs free-form captions and POPE needs yes/no probes -- so they run as
separate passes rather than being squeezed into the runner. Both share its
cache and its resume-safety.

**CHAIR** runs in captioning mode across the visual-degradation ladder, because
that is where it earns its place: it is the metric that detects the transition
from seeing to guessing, and the Resolution Threshold is defined as the dose
where it jumps more than 10 points over clean.

**POPE** runs on the object sub-type across all axes. Its accuracy/F1 measure
object hallucination; its yes-ratio, tracked against distractor count, doubles as
a sycophancy signal -- a model talked into agreeing by noisier context.
"""

from __future__ import annotations

import json
from collections import defaultdict

from gib.adapters.cache import ResponseCache
from gib.adapters.registry import get_adapter, is_simulated
from gib.config import RESULTS_DIR, VARIANTS_DIR, ensure_dirs
from gib.eval.metrics.chair import compute_chair, resolution_threshold
from gib.eval.metrics.pope import build_probes, score_pope, sycophancy_slope
from gib.eval.prompt import CAPTION_PROMPT
from gib.schema import stable_hash


def _load_variants(dataset: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(VARIANTS_DIR / dataset).to_pylist()


def _gt_objects_for(rows: list[dict]) -> dict[str, set[str]]:
    """image_id -> ground-truth object inventory.

    Prefers the real COCO instance index; falls back to the objects recoverable
    from the pairs themselves so the pass still runs on a synthetic build.
    """
    from gib.dataengine.sources.coco import load_instance_index

    index = load_instance_index()
    if index:
        return index

    inv: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        stem = r["pair_id"].rsplit("-", 2)[0]
        if r["answer_visual"]:
            inv[stem].add(r["answer_visual"])
    return dict(inv)


# --------------------------------------------------------------------------- #
# CHAIR
# --------------------------------------------------------------------------- #


def run_chair(
    model: str,
    *,
    dataset: str = "gib15k_eval.parquet",
    n_per_dose: int = 300,
    adapter_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Caption clean and degraded images, then score CHAIR at each dose."""
    ensure_dirs()
    rows = _load_variants(dataset)
    gt = _gt_objects_for(rows)

    adapter = get_adapter(model, **(adapter_kwargs or {}))
    if not adapter.capabilities.get("captioning"):
        raise RuntimeError(f"{model} does not support captioning mode")
    cache = ResponseCache()
    prompt_hash = stable_hash({"p": CAPTION_PROMPT, "mode": "caption"})

    # Clean, plus each Gaussian-blur dose. Grouping by dose is what makes the
    # threshold computable; a single pooled CHAIR would hide it entirely.
    buckets: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        if r["axis"] == "clean":
            buckets[0.0].append(r)
        elif r["axis"] == "visual" and r["degradation"] == "gaussian_blur":
            buckets[float(r["degradation_param"])].append(r)

    results: dict[float, dict] = {}
    candidate_pool = sorted({o for objs in gt.values() for o in objs})

    for dose in sorted(buckets):
        sample = sorted(buckets[dose], key=lambda r: r["variant_id"])[:n_per_dose]
        captions, inventories = [], []
        for r in sample:
            stem = r["pair_id"].rsplit("-", 2)[0]
            present = gt.get(stem, set())
            hit = cache.get(f"{model}:caption", r["variant_id"], prompt_hash, 0.0, 0)
            if hit is not None:
                cap = hit["answer"]
            else:
                cap = adapter.caption(
                    r["image_path"],
                    meta={
                        "variant_id": r["variant_id"], "axis": r["axis"],
                        "degradation": r["degradation"],
                        "degradation_param": r["degradation_param"],
                        "gt_objects": sorted(present),
                        "candidate_objects": candidate_pool,
                    },
                )
                cache.put(f"{model}:caption", r["variant_id"], prompt_hash, 0.0, 0,
                          {"answer": cap})
            captions.append(cap)
            inventories.append(present)

        res = compute_chair(captions, inventories)
        results[dose] = res.as_dict()
        if verbose:
            print(f"  [chair] sigma={dose:<5g} CHAIR_i={res.chair_i:.3f} "
                  f"CHAIR_s={res.chair_s:.3f}  (n={res.n_captions})")

    clean_chair = results.get(0.0, {}).get("chair_i", 0.0)
    by_dose = {d: v["chair_i"] for d, v in results.items() if d > 0}
    thr = resolution_threshold(by_dose, clean_chair)

    out = {
        "model": model,
        "simulated": is_simulated(model),
        "clean_chair_i": clean_chair,
        "by_dose": results,
        # None means the model never crossed the threshold across the tested
        # range -- reported as ">16" rather than imputed.
        "resolution_threshold_sigma": thr,
    }
    path = RESULTS_DIR / f"model={model}" / "chair.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    adapter.close()
    if verbose:
        print(f"[chair] {model}: resolution threshold sigma = {thr if thr is not None else '>max'}")
    return out


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #


def run_pope(
    model: str,
    *,
    dataset: str = "gib15k_eval.parquet",
    regimes: tuple[str, ...] = ("random", "popular", "adversarial"),
    per_image: int = 6,
    max_images: int = 500,
    adapter_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run POPE probes in all three negative-sampling regimes."""
    ensure_dirs()
    rows = _load_variants(dataset)
    gt = _gt_objects_for(rows)

    image_paths: dict[str, str] = {}
    for r in rows:
        if r["sub_type"] != "object":
            continue
        stem = r["pair_id"].rsplit("-", 2)[0]
        image_paths.setdefault(stem, r["image_path"])

    inventory = {k: v for k, v in gt.items() if k in image_paths}
    if max_images:
        keep = sorted(inventory)[:max_images]
        inventory = {k: inventory[k] for k in keep}

    adapter = get_adapter(model, **(adapter_kwargs or {}))
    cache = ResponseCache()
    out: dict = {"model": model, "simulated": is_simulated(model), "regimes": {}}

    for regime in regimes:
        probes = build_probes(inventory, image_paths, regime=regime, per_image=per_image)
        if not probes:
            continue
        answers, labels = [], []
        for p in probes:
            key = f"{p.image_id}:{p.obj}:{regime}"
            prompt_hash = stable_hash({"q": p.question, "mode": "pope"})
            hit = cache.get(f"{model}:pope", key, prompt_hash, 0.0, 0)
            if hit is not None:
                ans = hit["answer"]
            else:
                res = adapter.generate(
                    p.image_path, p.question, [],
                    meta={"variant_id": key, "axis": "clean",
                          "answer_visual": "yes" if p.label else "no",
                          "answer_textual": ""},
                )
                ans = res.answer
                cache.put(f"{model}:pope", key, prompt_hash, 0.0, 0, {"answer": ans})
            answers.append(ans)
            labels.append(p.label)

        res = score_pope(answers, labels, regime)
        out["regimes"][regime] = res.as_dict()
        if verbose:
            print(f"  [pope] {regime:<12} acc={res.accuracy:.3f} F1={res.f1:.3f} "
                  f"yes={res.yes_ratio:.3f}  (n={res.n})")

    # -- sycophancy: yes-ratio against distractor count ---------------------- #
    yes_by_dose = _yes_ratio_by_dose(model, rows, adapter, cache, verbose=verbose)
    if len(yes_by_dose) >= 2:
        out["yes_ratio_by_distractors"] = yes_by_dose
        out["sycophancy_slope"] = sycophancy_slope(yes_by_dose)
        if verbose:
            print(f"  [pope] sycophancy slope = {out['sycophancy_slope']:+.4f} "
                  f"(positive = talked into agreeing)")

    path = RESULTS_DIR / f"model={model}" / "pope.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    adapter.close()
    return out


def _yes_ratio_by_dose(model, rows, adapter, cache, *, per_dose=200, verbose=True) -> dict[int, float]:
    """Yes-ratio on existence probes as retrieval noise increases.

    Runs the probe *with the noisy passages attached*, which is the whole point:
    the question is whether irrelevant retrieved context pushes the model toward
    agreement independently of the image.
    """
    from gib.eval.metrics.pope import parse_yes_no
    from gib.eval.prompt import POPE_TEMPLATE

    by_dose: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["axis"] == "noise" and r["sub_type"] == "object":
            by_dose[int(r["n_distractors"])].append(r)

    out: dict[int, float] = {}
    for dose in sorted(by_dose):
        sample = sorted(by_dose[dose], key=lambda r: r["variant_id"])[:per_dose]
        yes = 0
        n = 0
        for r in sample:
            # Probe for the object the poisoned text asserts but the image lacks.
            target = r["answer_textual"] or r["answer_visual"]
            if not target:
                continue
            article = "an" if target[:1].lower() in "aeiou" else "a"
            q = POPE_TEMPLATE.format(article=article, obj=target)
            key = f"{r['variant_id']}:syco"
            prompt_hash = stable_hash({"q": q, "mode": "pope_noise"})
            hit = cache.get(f"{model}:pope", key, prompt_hash, 0.0, 0)
            if hit is not None:
                ans = hit["answer"]
            else:
                res = adapter.generate(
                    r["image_path"], q, list(r["retrieved_docs"] or []),
                    meta={"variant_id": key, "axis": "noise",
                          "n_distractors": dose,
                          "answer_visual": "no", "answer_textual": "yes"},
                )
                ans = res.answer
                cache.put(f"{model}:pope", key, prompt_hash, 0.0, 0, {"answer": ans})
            parsed = parse_yes_no(ans)
            n += 1
            yes += int(parsed is True)
        if n:
            out[dose] = yes / n
    return out
