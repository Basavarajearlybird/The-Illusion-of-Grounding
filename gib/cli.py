"""GIB-15K command-line interface.

Every layer is reachable here, and the Makefile is a thin wrapper over these
subcommands:

    python -m gib.cli dataset   --pairs 15000 --sources coco visual_genome
    python -m gib.cli eval      --model llava-1.6-7b
    python -m gib.cli interp    --model llava-1.6-7b
    python -m gib.cli human-sample
    python -m gib.cli human-score
    python -m gib.cli figures
    python -m gib.cli release
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_dataset(args) -> int:
    from gib.dataengine.build import build

    build(args.config, n_pairs=args.pairs, sources=args.sources)
    return 0


def cmd_models(args) -> int:
    from gib.adapters.registry import available_models, import_errors

    print("Available models:")
    for m in available_models():
        tag = "  [SIMULATED]" if m.startswith("sim-") else ""
        print(f"  {m}{tag}")
    errs = import_errors()
    if errs:
        print("\nAdapter modules that failed to import (missing optional deps):")
        for k, v in errs.items():
            print(f"  {k}: {v}")
    return 0


def cmd_eval(args) -> int:
    from gib.eval.runner import RunConfig, run

    cfg = RunConfig(
        model=args.model,
        dataset=args.dataset,
        axes=args.axes,
        sub_types=args.sub_types,
        limit=args.limit,
        n_samples=args.n_samples,
        temperature=args.temperature,
        judge_model=args.judge,
        offline_judge=args.offline_judge,
        allow_circular_judge=args.allow_circular_judge,
        proxy_model=args.proxy_model,
        measure_csr=not args.no_csr,
        measure_trust=not args.no_trust,
        adapter_kwargs=json.loads(args.adapter_kwargs) if args.adapter_kwargs else {},
    )
    run(cfg, resume=not args.no_resume)
    return 0


def cmd_probes(args) -> int:
    """CHAIR (captioning mode) and POPE (yes/no existence probing).

    Separate from `eval` because both need a different interaction mode from the
    main QA loop; they share its cache and resume-safety.
    """
    from gib.eval.probes import run_chair, run_pope

    kwargs = json.loads(args.adapter_kwargs) if args.adapter_kwargs else {}
    if args.metric in ("chair", "both"):
        run_chair(args.model, dataset=args.dataset, adapter_kwargs=kwargs)
    if args.metric in ("pope", "both"):
        run_pope(args.model, dataset=args.dataset, adapter_kwargs=kwargs)
    return 0


def cmd_interp(args) -> int:
    """Layer 3. Runs *after* the eval, because the probe set is selected using
    the Knowledge Inertia labels the eval produces."""
    import pyarrow.parquet as pq

    from gib.analysis import load as L
    from gib.config import VARIANTS_DIR
    from gib.interp.probe import run_probe, select_probe_set

    df = L.load_results(models=[args.model])
    variants = pq.read_table(VARIANTS_DIR / args.dataset).to_pylist()
    probe = select_probe_set(df.to_dict("records"), variants, n_conflict=args.n_conflict)
    print(f"[interp] probe set: {len(probe.conflict_ids)} conflict "
          f"({len(probe.inertia_ids)} inertia), {len(probe.twin_ids)} twins, "
          f"{len(probe.clean_ids)} clean controls")
    run_probe(args.model, probe, variants,
              adapter_kwargs=json.loads(args.adapter_kwargs) if args.adapter_kwargs else {})
    return 0


def cmd_human_sample(args) -> int:
    from gib.human.sample import build_study_sample, save_study, verify_identical

    items = build_study_sample(args.dataset, n_items=args.n_items)
    bad = verify_identical(items)
    if bad:
        print(f"[human] WARNING: {len(bad)} items failed the byte-identity check "
              f"against the dataset hashes; regenerate the dataset before running "
              f"the study. First few: {bad[:5]}")
    path = save_study(items)
    print(f"[human] {len(items)} items -> {path}")
    print("[human] serve with:  streamlit run gib/human/app.py")
    return 0


def cmd_human_simulate(args) -> int:
    """Write SIMULATED participant responses so Layer 4's scoring path can be
    exercised before recruitment. Never mixed with real responses."""
    from gib.human.sample import load_study
    from gib.human.simulate import simulate_responses

    simulate_responses(load_study(), n_participants=args.n_participants)
    return 0


def cmd_human_score(args) -> int:
    from gib.human.sample import load_study
    from gib.human.score import load_responses, save_summary, score_responses, summarize

    items = {i.variant_id: i for i in load_study()}
    responses = load_responses(allow_simulated=args.allow_simulated)
    if not responses:
        print("[human] no responses found; run the Streamlit app first "
              "(or `gib human-simulate` + `--allow-simulated` for a pipeline check)")
        return 1
    scored = score_responses(responses, items)
    summary = summarize(scored)
    path = save_summary(summary)
    print(f"[human] {summary.n_participants} participants, {summary.n_responses} responses")
    print(f"[human] overall accuracy {summary.overall_accuracy:.3f}, "
          f"Krippendorff α {summary.krippendorff_alpha:.3f}, "
          f"median RT {summary.median_rt_s:.1f}s")
    for axis, s in sorted(summary.by_axis.items()):
        print(f"           {axis:10s} {s['accuracy']:.3f}  (n={s['n']})")
    print(f"[human] -> {path}")
    return 0


def cmd_figures(args) -> int:
    from gib.analysis.run_all import run_all

    run_all(models=args.models)
    return 0


def cmd_release(args) -> int:
    from gib.dataengine.release import build_release

    build_release(version=args.version)
    return 0


def cmd_validate(args) -> int:
    from gib.dataengine.validate import validate_dataset

    report = validate_dataset(args.dataset)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_characterize(args) -> int:
    """Measured dataset characterisation for the datasheet + paper_numbers."""
    from gib.dataengine.characterize import run, to_markdown

    report = run(args.dataset)
    print(to_markdown(report))
    return 0


def cmd_paper_audit(args) -> int:
    """Check every provenance reference in the manuscript(s) against measured
    numbers. Non-zero exit gates a release."""
    from pathlib import Path

    from gib.analysis.audit import audit_paths
    from gib.analysis.provenance import write_provenance_csv

    paths = [Path(p) for p in (args.manuscript or [])]
    if not paths:
        # default: everything under paper/ that could carry numbers
        paths = sorted(Path("paper").glob("*.md")) if Path("paper").is_dir() else []
    if not paths:
        print("[paper-audit] no manuscript files given and paper/*.md is empty; nothing to check")
        return 0
    result, _ = audit_paths(paths, rel_tol=args.rel_tol)
    csv_path = write_provenance_csv()
    print(result.report())
    print(f"[paper-audit] provenance table: {csv_path}")
    return 0 if result.ok else 1


def cmd_render_paper(args) -> int:
    """Fill a manuscript template's {{provenance.key}} placeholders from measured
    numbers, so no quoted value can be stale."""
    from pathlib import Path

    from gib.analysis.audit import render_paths

    errors = render_paths(Path(args.template), Path(args.out))
    if errors:
        for e in errors:
            print(f"  UNRESOLVED: {e}")
        print(f"[render] wrote {args.out} with {len(errors)} unresolved placeholder(s)")
        return 1
    print(f"[render] {args.out} — all placeholders resolved from provenance")
    return 0


def cmd_croissant(args) -> int:
    """Emit and validate the Croissant metadata against the reference loader."""
    import json as _json

    from gib.config import RELEASE_DIR
    from gib.dataengine.release import build_croissant, validate_croissant

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    path = RELEASE_DIR / "croissant.json"
    path.write_text(_json.dumps(build_croissant(args.version), indent=2), encoding="utf-8")
    rep = validate_croissant(path)
    print(f"[croissant] {path}")
    print(f"[croissant] validator={rep['validator']}  ok={rep['ok']}  "
          f"fields={rep['n_fields']}")
    for e in rep.get("errors", []):
        print(f"  ERROR: {e}")
    return 0 if rep["ok"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gib", description="GIB-15K benchmark pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="Layer 1: build the benchmark")
    d.add_argument("--config", default="dataset.yaml")
    d.add_argument("--pairs", type=int, default=None, help="override n_pairs")
    d.add_argument("--sources", nargs="*", default=None,
                   help="coco visual_genome synthetic")
    d.set_defaults(func=cmd_dataset)

    m = sub.add_parser("models", help="list registered adapters")
    m.set_defaults(func=cmd_models)

    e = sub.add_parser("eval", help="Layer 2: run a model over the benchmark")
    e.add_argument("--model", required=True)
    e.add_argument("--dataset", default="gib15k_eval.parquet")
    e.add_argument("--axes", nargs="*",
                   default=["clean", "noise", "visual", "conflict", "combined"])
    e.add_argument("--sub-types", nargs="*", default=None, dest="sub_types")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--n-samples", type=int, default=10, dest="n_samples",
                   help="Track B sample count")
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--judge", default="claude-opus-4-8")
    e.add_argument("--offline-judge", action="store_true", dest="offline_judge",
                   help="degrade to the cascade when no judge backend is reachable")
    e.add_argument("--allow-circular-judge", action="store_true",
                   dest="allow_circular_judge")
    e.add_argument("--proxy-model", default=None, dest="proxy_model",
                   help="open model that scores CSR for Track B backends")
    e.add_argument("--no-csr", action="store_true")
    e.add_argument("--no-trust", action="store_true")
    e.add_argument("--no-resume", action="store_true")
    e.add_argument("--adapter-kwargs", default=None, dest="adapter_kwargs",
                   help='JSON, e.g. \'{"load_in_4bit": true}\'')
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("probes", help="Layer 2: CHAIR and POPE passes")
    pr.add_argument("--model", required=True)
    pr.add_argument("--dataset", default="gib15k_eval.parquet")
    pr.add_argument("--metric", choices=["chair", "pope", "both"], default="both")
    pr.add_argument("--adapter-kwargs", default=None, dest="adapter_kwargs")
    pr.set_defaults(func=cmd_probes)

    i = sub.add_parser("interp", help="Layer 3: attention probe (Track A models)")
    i.add_argument("--model", required=True)
    i.add_argument("--dataset", default="gib15k_eval.parquet")
    i.add_argument("--n-conflict", type=int, default=500, dest="n_conflict")
    i.add_argument("--adapter-kwargs", default=None, dest="adapter_kwargs")
    i.set_defaults(func=cmd_interp)

    hs = sub.add_parser("human-sample", help="Layer 4: build the human study sample")
    hs.add_argument("--dataset", default="gib15k_eval.parquet")
    hs.add_argument("--n-items", type=int, default=300, dest="n_items")
    hs.set_defaults(func=cmd_human_sample)

    hsim = sub.add_parser("human-simulate",
                          help="Layer 4: SIMULATED responses (pipeline check only)")
    hsim.add_argument("--n-participants", type=int, default=8, dest="n_participants")
    hsim.set_defaults(func=cmd_human_simulate)

    hc = sub.add_parser("human-score", help="Layer 4: score collected responses")
    hc.add_argument("--allow-simulated", action="store_true", dest="allow_simulated",
                    help="fall back to simulated responses when no real ones exist")
    hc.set_defaults(func=cmd_human_score)

    f = sub.add_parser("figures", help="Layer 5: regenerate every figure and table")
    f.add_argument("--models", nargs="*", default=None)
    f.set_defaults(func=cmd_figures)

    r = sub.add_parser("release", help="Layer 5: build the HF/GitHub release bundle")
    r.add_argument("--version", default="v1.0")
    r.set_defaults(func=cmd_release)

    v = sub.add_parser("validate", help="check the built dataset against its contract")
    v.add_argument("--dataset", default="gib15k_eval.parquet")
    v.set_defaults(func=cmd_validate)

    ch = sub.add_parser("characterize", help="measured dataset characterisation")
    ch.add_argument("--dataset", default="gib15k_eval.parquet")
    ch.set_defaults(func=cmd_characterize)

    cr = sub.add_parser("croissant", help="emit + validate Croissant metadata")
    cr.add_argument("--version", default="v1.0")
    cr.set_defaults(func=cmd_croissant)

    pa = sub.add_parser("paper-audit", help="check manuscript numbers against provenance")
    pa.add_argument("manuscript", nargs="*", help="manuscript files (default: paper/*.md)")
    pa.add_argument("--rel-tol", type=float, default=1e-3, dest="rel_tol")
    pa.set_defaults(func=cmd_paper_audit)

    rp = sub.add_parser("render-paper", help="fill a manuscript template from provenance")
    rp.add_argument("--template", required=True)
    rp.add_argument("--out", required=True)
    rp.set_defaults(func=cmd_render_paper)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
