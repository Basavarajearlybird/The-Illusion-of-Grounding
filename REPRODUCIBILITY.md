# Reproducibility

GIB-15K is built so that **every claim in the paper is produced by an automated,
re-runnable pipeline.** This document is the checklist a reviewer can walk.

## One-command reproduction

```bash
make install            # pinned deps: pip install -r requirements.lock
make dataset            # rebuild GIB-15K from COCO + Visual Genome
make validate           # contract check: 0 errors expected
make eval-open          # the open-weight model zoo (needs a GPU)
make figures            # every figure, table, and paper_numbers.json
make paper-audit        # verify every manuscript number against provenance
make release            # datasheet + Croissant + license bundle
```

For a machine with no GPU, no API keys, and no downloads:

```bash
make smoke              # the whole pipeline on synthetic data + simulated models
```

## The ML reproducibility checklist

**Code & data**
- [x] Source code released (CC-BY-4.0), one adapter/metric = one file.
- [x] Dataset regenerable from source; images verified by `image_sha256`.
- [x] Exact dependency versions pinned (`requirements.lock`).
- [x] Container with CUDA deps pinned (`Dockerfile`), tests run at build.

**Determinism**
- [x] All randomness derives from content hashes (`gib.seeds.derive_seed`); a
      partial re-run is byte-identical to a full run (enforced by a test).
- [x] Global seeds set for inference; `torch.use_deterministic_algorithms`.
- [x] `CUBLAS_WORKSPACE_CONFIG` pinned in the container.
- [x] Degraded images written once and content-hashed — no on-the-fly drift.

**Experimental rigour**
- [x] One frozen, versioned prompt template; its hash is in the cache key.
- [x] One frozen, versioned judge rubric; judge–human agreement reported.
- [x] Judge circularity blocked in code (`assert_no_circularity`).
- [x] Dual-track measurement (white/black-box) reported side by side, never
      pooled across tracks.
- [x] Statistical claims are tests with effect sizes, FDR-corrected across layers.

**Provenance**
- [x] Every result row logs git SHA, dataset hash, seed, prompt & judge version,
      adapter version.
- [x] `make figures` emits `paper_numbers.json` + `provenance.csv`: every scalar
      the paper can quote, addressable by key.
- [x] `make paper-audit` fails the build if any manuscript number disagrees with
      provenance. `make render-paper` fills the manuscript from provenance so a
      number *cannot* be typed by hand.
- [x] Claim language tracks the evidence (`gib.analysis.claims`): a weak effect
      renders "suggests", a strong one "demonstrates".

**Guards against silent failure**
- [x] `make dev-gate`: the whole pipeline must pass on a 1K subset before full
      compute is committed.
- [x] Integrity-verified, resumable downloads (`scripts/fetch.sh`) — a truncated
      archive is retried, never handed to the build.
- [x] Simulated adapters are watermarked, stamped, and warned about; their
      numbers cannot reach the paper unnoticed.

## What is NOT yet reproduced

- **Real model-zoo numbers.** The figures ship watermarked SIMULATED until an
  open-weight run is executed. The simulated adapters exist to verify the
  pipeline, not to produce results.
- **Semantic hard negatives** use TF-IDF in v1.0 (offline fallback). The
  `bge-large-en-v1.5` upgrade is an Axis-1-only re-run; the surgical negatives
  that carry the difficulty are encoder-independent.

## Environment the release was built in

See `requirements.lock`. Python 3.10–3.12 supported; CI runs all three.
Every build manifest (`data/gib15k/variants/manifest.json`,
`results/model=*/run_manifest.json`) records the git SHA and seed it was produced
under.
