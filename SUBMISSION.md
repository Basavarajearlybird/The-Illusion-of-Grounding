# GIB-15K — Submission package

This archive is a complete, self-contained snapshot of the GIB-15K project: the
full five-layer pipeline, the built benchmark, documentation, and a registered
experimental-design manuscript.

## What's included

| Path | What it is |
|---|---|
| `gib/` | All source code — the five-layer pipeline (~11k LOC) |
| `tests/` | 102 unit + integration tests |
| `configs/`, `scripts/` | Run configs, dataset-download scripts |
| `docs/`, `paper/` | Architecture notes, protocols, the manuscript template |
| `data/gib15k/variants/` | **The built GIB-15K benchmark** — 15,000 pairs → 325,000 variants as parquet, plus manifest and measured characterisation |
| `data/release/` | Datasheet (Gebru et al.), validated Croissant 1.0 metadata, license |
| `figures/` | Pipeline output — 17 figures + 9 tables + `paper_numbers.json` + `provenance.csv` |
| `README.md`, `LICENSE`, `CITATION.cff`, `REPRODUCIBILITY.md`, … | Project docs |

## What's deliberately excluded (and why)

| Excluded | Size | Reason |
|---|---|---|
| `data/raw/` (COCO + Visual Genome images) | ~68 GB | Under their own licenses; **not redistributable**. Regenerable — see below. |
| `data/images/degraded/` | ~12 GB | Regenerable from source by `make dataset`, verified byte-for-byte against the `image_sha256` column. |
| `.cache/`, `results/` | — | Regenerable pipeline caches and simulated-run outputs. |

This mirrors the project's licensing design: GIB-15K redistributes **annotations
plus transformation scripts**; the images are regenerated locally. See
`data/release/DATASHEET.md`.

## Reproduce from this archive

```bash
pip install -r requirements.txt          # core deps, CPU only, ~1 min
make smoke                               # full pipeline on synthetic data (no GPU, no API keys)
```

To rebuild the real benchmark and regenerate the images:

```bash
bash scripts/download_coco.sh && bash scripts/download_vg.sh
make dataset && make validate            # 0 contract errors expected
```

## Important honesty note

All **model-facing** numbers in `figures/` come from **simulated adapters** and
are watermarked `SIMULATED`. They exist to verify the pipeline end-to-end; they
are **not** scientific results. Producing real results requires running the
open-weight model zoo on a GPU (`make eval-open`). The benchmark *instrument* is
complete and validated; the model *evaluation* is the next step.

## Verification

```bash
make test        # 102 tests pass
make lint        # clean
make validate    # dataset contract: 0 errors
make paper-audit # every manuscript number matches provenance
```
