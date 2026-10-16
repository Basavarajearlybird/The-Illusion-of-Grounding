---
license: cc-by-4.0
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - multimodal
  - retrieval-augmented-generation
  - hallucination
  - robustness
  - benchmark
pretty_name: "GIB-15K: The Illusion of Grounding"
size_categories:
  - 10K<n<100K
configs:
  - config_name: clean
    data_files: "variants/axis=clean/*.parquet"
  - config_name: noise
    data_files: "variants/axis=noise/*.parquet"
  - config_name: visual
    data_files: "variants/axis=visual/*.parquet"
  - config_name: conflict
    data_files: "variants/axis=conflict/*.parquet"
  - config_name: combined
    data_files: "variants/axis=combined/*.parquet"
---

# GIB-15K — The Illusion of Grounding

A diagnostic benchmark for multimodal RAG under **retrieval noise**, **visual
ambiguity**, and **cross-modal conflict**.

## What makes it different

Adjacent benchmarks study these stressors individually. GIB-15K is the only one
that combines all three axes with **dosage–response curves**, an **internal
attention audit**, and a **human ceiling** in one instrument.

| Axis | Stressor | Dosages | Elite metric |
|---|---|---|---|
| 1 | Retrieval noise (hard negatives) | 1 / 3 / 5 / 10 / 20 distractors | Decay rate λ |
| 2 | Visual ambiguity | σ ∈ {{1,2,4,8,16}}; motion 5–40px; crop 50–5% | Resolution Threshold |
| 3 | Cross-modal conflict | image says A, text says B (+ counterfactual twin) | Trust Index |

## Splits

* `GIB-15K-eval` — the canonical split reported in the paper.
* `GIB-15K-full` — the complete variant superset.

The full cross-product is far larger than either and is deliberately **not**
evaluated; see the paper's dataset section.

## Key fields

`answer_visual` is the truth in the image; `answer_textual` is the falsehood
injected into the retrieved passages. Carrying both in the data is what makes the
failure taxonomy (Knowledge Inertia / Sensory Capture / Reasoning Collapse)
machine-classifiable at scale. `twin_id` links counterfactual pairs, so
counterfactual probing works out of the box.

## Reproducing

```bash
make dataset          # rebuild from COCO + Visual Genome
make eval MODEL=…     # run a model
make figures          # regenerate every figure and table
```

Images are regenerable rather than redistributed; verify with `image_sha256`.

## Citation

```bibtex
@misc{{gib15k,
  title  = {{The Illusion of Grounding: A Diagnostic Benchmark for Multimodal RAG
            under Retrieval Noise, Visual Ambiguity, and Cross-Modal Conflict}},
  year   = {{2026}}
}}
```

Licensed CC-BY 4.0. COCO and Visual Genome retain their original licences.
