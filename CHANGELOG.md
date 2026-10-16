# Changelog

All notable changes to GIB-15K are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow semver.

## [1.0.0] — 2026-07-30

First public release: the complete five-layer pipeline and the built benchmark.

### Dataset
- **15,000 image-QA pairs** from COCO 2017 + Visual Genome, balanced 5,000 each
  across object / attribute / relation sub-types.
- Three adversarial axes at controlled dosages: retrieval noise (1/3/5/10/20 hard
  negatives), visual ambiguity (Gaussian σ 1–16, motion 5–40px, crop 50–5%),
  cross-modal conflict with counterfactual twins.
- 325,000 total variants; canonical `GIB-15K-eval` split of 59,648 rows.
- Contract validation passes with 0 errors; measured hard-negative difficulty
  0.89 token-Jaccard; conflict twin yield 28%.
- Croissant 1.0 metadata validated against the reference `mlcroissant` loader.

### Pipeline
- **Layer 1** data engine (sources → scene-graph IR → template QA with
  round-trip validation → three adversarial generators).
- **Layer 2** evaluation engine: one `ModelAdapter` protocol, persistent response
  cache, dual-track (white/black-box) measurement, scoring cascade, automatic
  failure taxonomy, CHAIR/POPE/REFIND.
- **Layer 3** interpretability probe: Visual Attention Mass, paired Wilcoxon with
  effect size, Benjamini–Hochberg FDR.
- **Layer 4** human baseline: stratified sampling with verified byte-identity,
  Streamlit collection UI, Krippendorff's α.
- **Layer 5** analysis: 17 figures + 9 tables + `paper_numbers.json`, all from one
  `make figures`.

### Reproducibility
- Content-addressed determinism: partial re-runs are byte-identical to full runs.
- Integrity-verified, resumable downloads (`scripts/fetch.sh`).
- `make paper-audit` gate: every manuscript number checked against provenance.
- `make dev-gate`: the whole pipeline must pass on a 1K subset before full compute.
- Pinned dependency lock; environment captured in every manifest.

### Scholarship
- CC-BY-4.0 license, `CITATION.cff`, CI on Python 3.10–3.12, contribution guide,
  datasheet (Gebru et al.), issue/PR templates.

### Known limitations
- Semantic hard negatives currently use TF-IDF (offline fallback); upgrade to
  `bge-large-en-v1.5` is an Axis-1-only re-run.
- Model evaluation figures ship watermarked as SIMULATED until a real model zoo
  run is executed; the simulated adapters exist to verify the pipeline, not to
  produce results.
