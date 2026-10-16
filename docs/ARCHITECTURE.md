# GIB-15K — implementation notes

Where the code departs from, sharpens, or defends the architecture document.
Read the [README](../README.md) first for the overall shape.

---

## Design decisions worth defending in the paper

### 1. Counterfactual twins are *mined*, not *rewritten*

The architecture specifies that every conflict pair is generated as a
counterfactual twin, with the flip built into the dataset rather than the
evaluation logic. That is the right call and it is implemented — but it needs one
addition the design doc leaves implicit.

For an item where the image shows A and the text asserts B, the true
counterfactual is an image showing **B** with text asserting **A**. No amount of
text rewriting produces a photograph of an orange from a photograph of an apple.
So `conflict.plan_conflicts` mines the twin from the corpus: a real pair of the
same sub-type whose visual answer is B and whose own injected falsehood is A.
Both rows share a `twin_id`, and `validate.py` asserts the flip is exact in both
directions.

Items with no available partner are still emitted — they are valid conflict
items — but carry an empty `twin_id`, and `analysis.load.twin_table` uses only
genuinely paired rows. **Reporting the twinned subset rather than treating
unpaired items as twins is what keeps the symmetry score honest.**

Twin yield scales with corpus size (~4 % at 90 pairs, ~22 % at 1,200, higher at
15,000), since a partner becomes more likely as the answer vocabulary fills in.

### 2. The eval split preserves twin pairs

Stratified sampling initially tore twins apart: a pair survived only if *both*
halves were independently drawn, which orphaned ~90 % of them. The `twin_id`
column would still have been populated, so the loss would not have been obvious —
counterfactual symmetry would simply have been uncomputable on the canonical
split.

`build.stratified_eval_split` now pulls in the partner of any sampled twin. This
is why the eval split's row count slightly exceeds its nominal target.

### 3. Sensory Capture is the one inferential taxonomy label

Correct / Knowledge Inertia / Reasoning Collapse fall straight out of two judge
verdicts. Sensory Capture does not — it is defined as *following a degraded
percept over correct text*, which is only meaningful where a degraded percept
exists. So it is assigned **only** on the visual and combined axes, and only when
the answer is not a refusal.

That last condition matters: a model that says "I can't tell" under σ=16 is
behaving correctly about its own uncertainty. Counting that as Sensory Capture
would invert the finding.

### 4. The Trust Index is normalised, not the raw ratio

The proposal defines `TI = P(text-answer) / P(image-answer)`. That is unbounded —
it runs to infinity as the visual answer's probability approaches zero, which
makes it unusable as a figure axis and undefined at the most interesting items.

`eval/trust.py` computes both and plots the normalised form,
`P(text) / [P(text) + P(visual)]`, bounded in [0, 1] with 0.5 as the neutral
point. The raw ratio is retained in `trust_raw_ratio` for continuity with the
proposal.

Track A scores are **length-normalised** before comparison: "apple" and "orange"
tokenise to different lengths, and an unnormalised sequence probability would
confound modality preference with answer length.

### 5. Track B ignores off-manifold samples

The empirical Trust Index is computed over samples matching one of the two
candidate answers. Samples matching neither — refusals, collapse — would drag the
ratio toward 0.5 and mimic balance where there is none. Their share is
recoverable from `n_samples` and reported separately.

### 6. The Track B matcher skips the judge

The Track B Trust Index matches N sampled answers against two candidates using
the cascade **without** its judge stage. Judging N×2 strings per item would
multiply judge calls roughly 20-fold across the corpus for a marginal gain, since
sampled answers to these items are overwhelmingly short spans that exact and
alias matching already resolve.

The consequence — a Track B trust index is very slightly conservative relative to
a Track A one — is small but real, and belongs in a footnote rather than being
left for a reader to discover.

### 7. Judge circularity is enforced on model *family*

`assert_no_circularity` matches on family prefix, not exact model id. `gpt-4o`
judging `gpt-4o-mini` is the same conflict of interest, and an exact-match check
would wave it through. Overriding requires `--allow-circular-judge` **and**
reporting judge–human agreement for the affected models.

### 8. `is_correct` means "matches the image"

Throughout the codebase, correctness is agreement with `answer_visual`. On the
conflict axis this means a "correct" answer is one that **resists** the text.
That is the intended semantics — the benchmark measures grounding — but it makes
conflict-axis accuracy non-comparable to ordinary VQA accuracy, and the paper
should say so explicitly.

### 9. ECE is never pooled across tracks

Track A confidence is a length-normalised sequence probability; Track B
confidence is self-consistency across samples. These are different instruments.
Averaging them would produce a number belonging to neither, so `compute_robustness`
keeps `track` as a column and the tables report it.

### 10. CSR on Track B is a proxy measurement

REFIND/CSR needs token probabilities. For closed models an open model re-scores
the closed model's emitted string — a standard, citable workaround, but it
measures *the proxy's* context sensitivity toward that output. Those rows are
stamped `csr_proxy=True` and labelled in every figure. This is a weaker claim and
is not presented as anything else.

---

## Two silent failure modes, and the guards against them

The architecture flags both. Each is handled in code rather than by discipline.

**Prompt-format sensitivity.** One canonical template, `rag_prompt_v1.txt`,
frozen and released. `PROMPT_VERSION` is logged on every result row, and the
**cache key includes the prompt hash** — so editing the template correctly
invalidates the cache rather than silently serving responses generated under the
old framing. Passages are numbered so `gt_doc_position` is actually perceivable
by the model, which is what makes the position-bias ablation meaningful.

**Judge circularity.** See §7 above.

---

## Determinism

Every random choice derives from a SHA-256 of stable identifiers
(`seeds.derive_seed`), never from a global RNG whose state depends on iteration
order. Consequences:

- a partial re-run produces byte-identical variants to a full run;
- generating variant 5,000 does not depend on having generated variants 1–4,999;
- `motion_angle` is randomised per item yet exactly reproducible, so blur
  direction is neither a hidden constant nor a source of drift.

Degraded images are encoded once, hashed on the encoded bytes, and written only
if absent. The eval layer never degrades on the fly.

---

## Offline mode

The full pipeline runs with no downloads, no GPU, and no API keys:

| Real | Offline substitute | Guard |
|---|---|---|
| COCO + Visual Genome | `sources/synthetic.py` — procedurally rendered scenes with exact scene graphs | documented as a fixture in the datasheet |
| `bge-large-en-v1.5` | TF-IDF, then hashed bag-of-words | backend recorded in the build manifest |
| FAISS | chunked exact numpy search | backend recorded in the build manifest |
| LLaVA / Qwen / GPT-4o / Gemini | `adapters/simulated.py` | `simulated=True` on every row; watermark on every figure |
| LLM judge | cascade-only, latched after first failure | `judge.unavailable_hits` in the run manifest |
| Human participants | `human/simulate.py` | separate file; requires `--allow-simulated` |

The substitutions degrade *quality*, never *contract*. Every schema, key, and
code path exercised offline is the one used in the real build — which is the
point: it is how the pipeline is tested before the expensive resources exist.

**Fallbacks are recorded, never silent.** A build that used TF-IDF instead of
bge-large says so in `manifest.json`; a run whose judge was unreachable says so
in the run manifest.

---

## Cost control

The full cross-product of 15,000 pairs × 5 noise doses × 13 visual settings ×
conflict twins is far larger than needed. Two splits:

- **`GIB-15K-eval`** — stratified by axis × dose × sub_type, with a floor of 50
  items per cell (cells below it are taken whole rather than shrunk), plus twin
  completion. This is what the paper reports.
- **`GIB-15K-full`** — the superset, released for downstream users.

Open models run the full eval split; closed models run a stratified 3–5 K subset
at N=10 samples. The response cache means a re-run, a crash recovery, or a
re-analysis costs nothing.

---

## Extending

**Adding a model** — one file:

```python
from gib.adapters.hf_vlm import HFVLMAdapter
from gib.adapters.registry import register_adapter

class MyVLM(HFVLMAdapter):
    name, model_id = "my-vlm", "org/my-vlm"
    def format_prompt(self, text: str) -> str:
        return f"<|user|><image>\n{text}<|assistant|>"

register_adapter("my-vlm")(lambda **kw: MyVLM(**kw))
```

Track A instrumentation — Trust Index, CSR, VAM — comes for free, because all
three are built on `_teacher_forced_logprobs` and on locating the visual-token
span via the model's own `image_token_index`.

**Adding a stressor axis** — add the value to `schema.Axis`, write a generator
emitting `Variant` rows, call it from `dataengine.build.build_variants`. The eval
engine is axis-agnostic; only figures that name the axis explicitly need
touching.

**Adding a metric** — drop it in `eval/metrics/`, surface it as a column in
`runner.evaluate_row` or as a separate pass in `eval/probes.py` if it needs a
different interaction mode.
