# Paper requirements → implementation map

Every requirement from the project brief, and the code and artifact that
satisfies it. Reviewers of benchmark papers reward reproducibility above almost
everything else, so each row names the command that regenerates the claim.

---

## Project requirements

| # | Requirement | Implementation | Artifact |
|---|---|---|---|
| 1 | 15,000 image–QA pairs from COCO / Visual Genome / WikiData | `dataengine/sources/{coco,visual_genome,wikidata}.py` | `data/gib15k/clean_pairs.parquet` |
| 2 | Label every pair by sub-type: object / attribute / relation | `schema.SubType`; sources chosen so each sub-type has a native annotation source | `table8_dataset_composition` |
| 3 | Three adversarial variants per pair | `hardneg.py`, `degrade.py`, `conflict.py` | `variants/axis=*/data.parquet` |
| 4 | Release as GIB-15K, CC-BY, GitHub + HF | `dataengine/release.py` | `data/release/` — datasheet, Croissant, HF card, LICENSE |

## The Stress-Test Triad

| Requirement | Implementation | Artifact |
|---|---|---|
| **A.1** Dosage–response for each stressor | Every axis is dose-parameterised; each dose is a separate row keyed to the same `pair_id` | figs 1–4 |
| **A.2** Hard negatives, not random text | Two-stage: FAISS/embedding neighbours **+ surgical single-fact mutation** of the ground-truth passage (`hardneg.build_pools`) | — |
| **A.3** Decay Rate, 1→20 distractors | `metrics/robustness.fit_decay`, weighted by per-dose n | `fig1_decay_curves`, `table6_decay_rates` |
| **A.4** Gaussian noise, motion blur, low-resource cropping | `degrade.py` — crop keyed to the **answer object's bbox**, not the image centre | `fig2_resolution_thresholds` |
| **A.5** Resolution Threshold | `metrics/chair.resolution_threshold` — first dose where CHAIR jumps >10 pts | `fig3_chair_resolution_threshold` |
| **A.6** Cross-modal conflict, image A vs text B | `conflict.py`, entity swaps via the WikiData sibling graph | `axis=conflict` |
| **A.7** Trust Index | `eval/trust.py`, dual-track | `fig5_conflict_matrix`, `table4` |

## Failure taxonomy

| Requirement | Implementation | Artifact |
|---|---|---|
| **B.1** Formal definition table | `eval/taxonomy.TAXONOMY_DEFINITIONS` | `table2_failure_taxonomy` |
| **B.2** Sensory Capture | Visual/combined axes only, refusals excluded | `fig9_taxonomy_by_axis` |
| **B.3** Knowledge Inertia | Answer matches `answer_textual` — the injected falsehood | `table3_taxonomy_proportions` |
| **B.4** Reasoning Collapse | Matches neither modality | `fig9` |

Assigned automatically from two judge verdicts, so proportions are reported over
the **whole corpus** rather than a hand-labelled subset.

## Model Zoo

| Requirement | Implementation |
|---|---|
| **C.2** Open-source SOTA | `llava-1.6-7b`, `llava-1.6-13b`, `qwen-vl-chat`, `qwen2.5-vl-7b` |
| **C.3** Closed-source giants | `gpt-4o`, `gemini-1.5-pro` (+ `gpt-4o-mini`, `gemini-2.0-flash` frontier tier) |
| **J.3** ≥ 5 SOTA models, open vs. closed | 8 registered — `make models` |

## Mathematical rigor

| Requirement | Implementation | Artifact |
|---|---|---|
| **D** ℛ = (Acc_adv / Acc_clean) × (1 − Uncertainty Error) | `metrics/robustness.robustness_score` | `table5_robustness`, `fig8` |
| **D.1** Penalise confident-but-wrong | Uncertainty Error := **ECE**; `overconfidence_rate` reported alongside | `fig10_calibration` |

## Tables & visuals

| Requirement | Artifact |
|---|---|
| **E.1** Vulnerability Heatmap (blur × noise) | `fig4_vulnerability_heatmap` |
| **E.2** Conflict Matrix, Text- vs Image-Biased | `fig5_conflict_matrix`, `table4` |
| **E.3** Hallucination Gallery | `fig18_hallucination_gallery` + `hallucination_gallery.json` |
| **H** Pareto Frontier of Robustness | `fig7_pareto_frontier` |
| **I** Counterfactual probing | `fig6_counterfactual_symmetry` |

## Human study & interpretability

| Requirement | Implementation | Artifact |
|---|---|---|
| **F.1** 5–10 people, same adverse questions | `human/sample.py` + `human/app.py`; byte-identity **verified** by re-hashing | `study_items.json` |
| **F.2** Reasoning Gap | Scored with the *same* judge cascade as models | `fig11_human_gap`, `table7` |
| **A** (brief) Record accuracy **and time** | Per-item response time in the Streamlit app | `median_rt_s` |
| **G.2** Attention maps under conflict | `interp/hooks.py` — VAM per layer/head | `fig13`, `fig17` |
| **G.3** Attention ignores visual tokens, latches onto noisy text | Paired Wilcoxon + effect size + BH-corrected per-layer tests | `fig14`, `fig15` |
| **B** (brief) 2–3 panels per failure mode | `interp/figures.py` — 4 panels | `fig13`–`fig16` |

## Metric-market fit — CHAIR, POPE, REFIND

| Metric | Where applied | Implementation | Artifact |
|---|---|---|---|
| **CHAIR** | Visual axis, captioning mode | `metrics/chair.py`, driven by `eval/probes.run_chair` | `fig3`, `table9` |
| **POPE** | Object sub-type, all axes; yes-ratio as sycophancy | `metrics/pope.py`, `eval/probes.run_pope` | `table9` |
| **REFIND / CSR** | Noise + conflict axes | `metrics/refind.py`, native on Track A, proxy-scored on Track B | `table4`, `table9` |

## "The most important" (J)

| # | Requirement | Where |
|---|---|---|
| J.1 | Define the Grounding Illusion — models as stochastic parrots of retrieved text | README opening; operationalised as the Knowledge Inertia rate + CSR on the injected falsehood |
| J.2 | The Triad at systematic levels | Three dose-parameterised axes + the combined sub-grid |
| J.3 | ≥ 5 SOTA models, open vs. closed | 8 adapters |
| J.4 | Formal Sensory Capture vs. Knowledge Inertia | `table2`; the distinction is machine-computable, not editorial |
| J.5 | Human baseline as the gold standard | Layer 4 |
| J.6 | Attention heatmaps of the tug-of-war | Layer 3 |

---

## Regenerating everything

```bash
make dataset      # Layer 1
make eval-open    # Layer 2, open tier
make eval-closed  # Layer 2, API tier
make probes MODEL=…   # CHAIR + POPE
make interp MODEL=…   # Layer 3
make human-sample && make human-app && make human-score   # Layer 4
make figures      # Layer 5 — every figure, table, and paper_numbers.json
make release
```

`figures/paper_numbers.json` contains every scalar the manuscript quotes, so no
number in the text is transcribed by hand.

## Claims this codebase will NOT let you make

- **Numbers from `sim-*` adapters.** Watermarked, stamped `simulated=True`, and
  warned about at run time.
- **A model grading its own outputs.** `assert_no_circularity` raises, matching
  on model family.
- **Counterfactual symmetry over unpaired items.** Only genuinely twinned rows
  enter `twin_table`.
- **Track A and Track B calibration pooled.** `track` stays a column throughout.
- **CSR on closed models presented as native.** Proxy rows are stamped and
  labelled.
- **A human baseline from simulated participants.** Separate file; requires an
  explicit `--allow-simulated`.
