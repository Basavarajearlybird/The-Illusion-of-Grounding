# GIB-15K — The Illusion of Grounding

**A Diagnostic Benchmark for Multimodal RAG under Retrieval Noise, Visual Ambiguity, and Cross-Modal Conflict**

[![CI](https://img.shields.io/badge/CI-py3.10--3.12-brightgreen)](.github/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-102%20passing-brightgreen)](tests/)
[![license](https://img.shields.io/badge/license-CC--BY--4.0-blue)](LICENSE)
[![dataset](https://img.shields.io/badge/dataset-15K%20pairs%20%C2%B7%20325K%20variants-orange)](data/release/)
[![croissant](https://img.shields.io/badge/Croissant-1.0%20validated-green)](data/release/croissant.json)

15,000 image–QA pairs from COCO, Visual Genome, and WikiData, labelled by
hallucination sub-type, each carrying three adversarial variants at controlled
dosages. Released CC-BY on GitHub and the HuggingFace Hub.

> **Reproducibility at a glance.** Every number in the paper is pulled from
> `figures/paper_numbers.json` at render time and re-checked by `make paper-audit`
> — none is typed by hand. All randomness is content-addressed, so a partial
> re-run is byte-identical to a full one. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## The claim

Multimodal RAG systems look grounded. They cite a passage, describe an image, and
answer fluently. GIB-15K asks whether that appearance survives contact with three
stressors applied at measured doses — and finds the conditions under which a model
stops reasoning over the image and becomes a **stochastic parrot of the retrieved
text**. We call that failure the **Grounding Illusion**.

Adjacent benchmarks study these stressors individually. GIB-15K is the only
instrument that combines all three axes with **dosage–response curves**, an
**internal attention audit**, and a **human ceiling**.

| Axis | Stressor | Dosages | Elite metric |
|---|---|---|---|
| **1** | Retrieval noise — hard negatives, semantically near but factually wrong | 1 / 3 / 5 / 10 / 20 distractors | **Decay rate λ** |
| **2** | Visual ambiguity — Gaussian blur, motion blur, low-resource cropping | σ ∈ {1,2,4,8,16}; 5–40 px; 50→5 % of object visible | **Resolution Threshold** |
| **3** | Cross-modal conflict — image says A, text says B, plus a counterfactual twin | n/a | **Trust Index** |

### The failure taxonomy

Failures are not reported as "wrong". They are classified, automatically, over
the entire corpus:

| Mode | Meaning |
|---|---|
| **Knowledge Inertia** | Ignored the image, followed the poisoned text |
| **Sensory Capture** | Ignored correct text, followed a degraded percept |
| **Reasoning Collapse** | Matched neither modality; incoherent or off-topic |

This is machine-classifiable because the dataset schema carries **both**
`answer_visual` (the truth in the image) and `answer_textual` (the falsehood
injected into the passages). Encoding the falsehood in the *data* is what lets
taxonomy proportions be reported across 60K items instead of a hand-labelled
subset.

---

## Quickstart

```bash
make install          # core deps, CPU only, ~1 minute
make smoke            # full offline pipeline: dataset → eval → figures → release
```

`make smoke` runs end to end with **no downloads, no GPU, and no API keys**,
using a synthetic scene source and simulated adapters. It exercises every stage
of the real pipeline; its outputs are watermarked `SIMULATED DATA` and are not
results. See [Simulated adapters](#simulated-adapters).

To build the real benchmark:

```bash
make install-heavy                                   # + torch, transformers, faiss
bash scripts/download_coco.sh && bash scripts/download_vg.sh
make dataset                                         # Layer 1  (~1–2 days wall-clock)
make validate
make eval-open                                       # Layer 2  (1× A100 80 GB)
make eval-closed                                     # Layer 2  (API tier, cached)
make interp MODEL=llava-1.6-7b                       # Layer 3
make human-sample && make human-app                  # Layer 4
make figures                                         # Layer 5  — every figure and table
make release
```

---

## Architecture

Five decoupled layers connected by versioned data contracts (parquet files with
fixed schemas). Any layer re-runs independently without invalidating downstream
results, and the entire figure set regenerates from one command.

```
  Layer 1  Data Engine  ──────►  data/gib15k/variants/*.parquet
                                       │
  Layer 2  Evaluation Engine  ─────────┼──────►  results/model=*/axis=*/*.parquet
                                       │              │
  Layer 3  Interpretability Probe  ────┤              │   (branches off, on Track A models)
  Layer 4  Human Baseline  ────────────┘              │   (branches off, runs in parallel)
                                                      ▼
  Layer 5  Analysis & Release  ──────────────►  figures/ + tables/ + paper_numbers.json
```

```
gib/
  schema.py            the data contract every layer reads and writes
  seeds.py             content-addressed determinism
  dataengine/          Layer 1 — sources, QA generation, the three adversarial generators
  adapters/            Layer 2 — the model zoo behind one Protocol
  eval/                Layer 2 — scoring cascade, judge, taxonomy, metrics, runner
  interp/              Layer 3 — attention hooks, VAM, paired statistics
  human/               Layer 4 — sampling, Streamlit UI, agreement statistics
  analysis/            Layer 5 — DuckDB aggregation, figures, tables, gallery
```

### Layer 1 — the Data Engine

Sources normalise into one scene-graph IR, so the mutators operate *structurally*
rather than by regex over prose. Questions are template-first (verifiable by
construction), LLM-paraphrased second, and every pair passes **round-trip
validation**: an independent solver re-answers the question from the scene graph
alone, and mismatches are discarded. This is what filters ambiguous items — "what
colour is the chair?" in a two-chair scene has two admissible answers and never
reaches the corpus.

The adversarial layer is a pure function `(clean_pair, config) → variant`,
deterministic under a fixed seed. Hard negatives combine embedding nearest
neighbours with **surgical corruption** — mutating the ground-truth passage so it
differs by exactly one fact, which is the hardest possible negative: identical
register, identical topic, one flipped fact. Entity swaps use the WikiData
sibling graph, so `apple → orange`, never `apple → helicopter`.

**Counterfactual twins are built into the dataset, not the evaluation logic.**
Because no rewriting can turn a photo of an apple into a photo of an orange, each
twin is *mined from the corpus*: a real pair whose visual answer is the other's
falsehood. Both rows share a `twin_id`, and the validator checks the flip is
exact. Items with no partner are emitted with an empty `twin_id` and excluded
from symmetry analyses rather than quietly counted as twins.

### Layer 2 — the Evaluation Engine

One `ModelAdapter` Protocol; adding a model is a ~40-line file. Open models run
through Transformers (LLaVA-1.6 7B/13B, Qwen-VL, Qwen2.5-VL); closed models
through API adapters with exponential backoff, token-bucket rate limiting, and a
**persistent response cache** keyed on
`(model, variant_id, prompt_hash, temperature, sample_idx)`. The cache is what
makes a 60K-item benchmark financially survivable and what lets a crashed run
resume for free.

**The dual-track measurement design.** The Trust Index and Robustness Score both
assume token probabilities — and GPT-4o exposes only truncated top-k logprobs
while Gemini exposes effectively none. Rather than hiding this as a limitation,
the architecture reports two tracks side by side and claims the resolution as a
methodological contribution:

- **Track A (white-box).** Constrained scoring under teacher forcing.
  `TI = P(textual) / [P(textual) + P(visual)]`; TI > 0.5 is text-biased.
- **Track B (black-box).** N = 8–10 samples at T ≈ 0.7. TI is the empirical
  fraction matching the textual answer; confidence is self-consistency.

Both tracks emit the same columns, so every figure renders for every model. The
tracks are never pooled into a single calibration number — they are different
instruments.

Answers are scored by a cascade — normalised exact match → alias match → pinned
**LLM-as-judge** with a frozen rubric — run **twice**, once against
`answer_visual` and once against `answer_textual`. Those two verdicts are exactly
what the taxonomy classifier consumes.

### Layer 3 — the Interpretability Probe

In LLaVA-style architectures image patches enter the LM as projected visual
tokens, so cross-modal attention is the decoder's attention over the visual-token
span. Define

> **VAM** = Σ attention to visual-token positions / Σ attention to all positions

per layer and head, averaged over answer-generation steps. The headline claim —
*during Knowledge Inertia, visual attention collapses* — is tested as a **paired
Wilcoxon signed-rank test on matched items** with a reported effect size, not
asserted from a suggestive figure. Per-layer tests carry Benjamini–Hochberg FDR
correction, because testing 32 layers at α = .05 is expected to yield ~1.6
spurious "significant" layers.

Runs on a curated probe set (~500 conflict items + twins + matched clean
controls) selected *after* the eval, since selecting Knowledge Inertia items
requires the eval's labels.

### Layer 4 — the Human Baseline

~300 items stratified by axis × dose × sub_type, drawn from the exact split the
models saw. Byte-identity is **verified**, not assumed — every served image is
re-hashed against `image_sha256`. Humans are scored with the *identical* judge
cascade as the models; that symmetry is what makes the gap legitimate.

The stratification is the design: it shows where the gap opens (conflict) **and
where it honestly narrows** (extreme blur). Reporting the narrowing case is what
makes the study credible.

### Layer 5 — Analysis & Release

`make figures` emits the complete artifact set plus `paper_numbers.json`
containing every scalar the manuscript quotes, so no number is transcribed by
hand.

| Figure | What it shows |
|---|---|
| Decay curves | Accuracy vs. distractors, with fitted λ per model |
| Resolution thresholds | Accuracy + CHAIR vs. σ / kernel / crop % |
| **Vulnerability Heatmap** | Δ-accuracy over the blur × noise grid |
| **Conflict Matrix** | Trust Index, taxonomy shares, Text/Image-Biased verdict |
| Counterfactual symmetry | Is the text bias symmetric, or object-dependent? |
| **Pareto Frontier** | Clean accuracy vs. Robustness ℛ |
| Calibration | Reliability diagrams — the "confident but wrong" pathology |
| Human gap | The Reasoning Gap per stressor |
| Attention panels | Layer-wise VAM, the collapse test, head-level map |
| **Hallucination Gallery** | Auto-mined failures shared across ≥ 3 models |

---

## Metric-market fit

| Metric | Where it applies | Why here |
|---|---|---|
| **CHAIR** | Visual axis, captioning mode | Detects the transition from seeing to guessing — the Resolution Threshold is where CHAIR jumps > 10 pts |
| **POPE** | Object sub-type, all axes | Yes/no existence probing; its yes-ratio doubles as a **sycophancy** signal under retrieval noise |
| **REFIND / CSR** | Noise + conflict axes | Token-level context sensitivity. High CSR on tokens matching the injected falsehood is *direct, quantitative evidence of Knowledge Inertia* |

CSR needs token probabilities, so it runs natively on Track A and via a **proxy
scorer** on Track B. Proxy rows are stamped `csr_proxy=True` and labelled in
every figure — the proxy measures its own context sensitivity toward the closed
model's output, which is a weaker claim and is not presented as anything else.

**The Robustness Score.**

> ℛ = (Acc_adverse / Acc_clean) × (1 − ECE)

"Uncertainty Error" is operationalised as **Expected Calibration Error**, which
is precisely what the formula is designed to punish: confident-but-wrong *is*
high ECE. ℛ is a ratio, so it must always be reported alongside clean accuracy —
which is exactly what the Pareto frontier plots.

---

## Simulated adapters

`sim-*` models are **not models**. They are stressor-responsive fixtures that let
the pipeline, the judge cascade, and every figure function be developed and
tested before a GPU or an API key exists — the "tracer bullet" the build order
calls for.

Three safeguards make them impossible to mistake for results:

1. every result row is stamped `simulated=True`;
2. every figure rendered from them carries a `SIMULATED DATA / NOT A RESULT`
   watermark, applied from the *data* rather than from a caller-passed flag;
3. the runner prints a warning on every simulated run.

**No number produced by a `sim-*` adapter belongs in the paper.**

---

## Reproducibility

- All randomness is **content-addressed** — seeds derive from stable identifiers,
  so a partial re-run is byte-identical to a full one.
- Degraded images are written once and content-hashed; the eval never degrades on
  the fly, so results cannot drift.
- Every result row logs git SHA, dataset hash, seed, prompt version, judge model,
  and adapter version.
- The prompt template and judge rubric are **frozen and versioned**; both are
  released. The cache key includes the prompt hash, so editing the template
  correctly invalidates it.
- **Judge circularity is blocked in code** (`assert_no_circularity`), matching on
  model *family* — `gpt-4o` judging `gpt-4o-mini` is the same conflict of
  interest, and an exact-match check would wave it through.

### The paper-audit gate — no number is typed by hand

`make figures` emits `paper_numbers.json` (every scalar the paper can quote) and
a flat `provenance.csv`. The manuscript is a **template** whose numbers are
provenance placeholders:

```bash
make render-paper   # fills paper/gib15k.md from provenance — a number cannot be stale
make paper-audit    # fails the build if any reference is unresolved or drifts
```

`make release` depends on `paper-audit`, so a release cannot ship a manuscript
that disagrees with what the pipeline measured. The claim **verbs** are generated
from the statistics too (`gib.analysis.claims`): a strong effect renders
"demonstrate", a wrong-signed or null effect renders "do not support" — the
language tracks the evidence.

### Safeguards before you spend compute

```bash
make dev-gate    # the whole pipeline must pass on a 1K subset before scaling
make validate    # dataset contract check (0 errors expected)
make test        # 102 unit + integration tests
make smoke       # full offline end-to-end run (no GPU, no API keys)
```

Downloads are integrity-verified and resumable (`scripts/fetch.sh`): a truncated
archive from a dropped connection is retried, never handed to the build.

---

## Compute envelope

| Component | Requirement | Notes |
|---|---|---|
| Data engine | CPU + 1 small GPU | Embedding + FAISS; OpenCV is CPU-bound; ~1–2 days |
| Open-model eval | 1× A100 80 GB (or 2× 4090) | LLaVA-13B is the ceiling; vLLM batching |
| Interpretability | Same GPU, hooks on | Probe set only (~1.5 K items) — hours |
| Closed-model eval | API budget | Stratified 3–5 K × N=10 × 2 models; cache makes it one-time |
| Human study | None | Streamlit on a laptop |

Closed models run a **stratified 3–5 K subset**, stratified by axis × dose ×
sub_type so every heatmap cell keeps n ≥ 50. Expect low hundreds of dollars, paid
once.

---

## Licence

Code and annotations: **CC-BY 4.0**. COCO and Visual Genome retain their original
licences — GIB-15K redistributes annotations plus transformation scripts, and
degraded images are regenerable from source and verifiable against
`image_sha256`. This sidesteps most redistribution questions.

## Citation

```bibtex
@misc{gib15k,
  title  = {The Illusion of Grounding: A Diagnostic Benchmark for Multimodal RAG
            under Retrieval Noise, Visual Ambiguity, and Cross-Modal Conflict},
  year   = {2026}
}
```
