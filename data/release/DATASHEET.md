# Datasheet for GIB-15K

Following Gebru et al., *Datasheets for Datasets*.

## Motivation

**For what purpose was the dataset created?**
To measure whether multimodal RAG systems are genuinely *grounded* or merely
fluent — what we call the **Grounding Illusion**, in which a model appears to
reason over an image while in fact acting as a stochastic parrot of the retrieved
text. Existing benchmarks test these stressors in isolation. GIB-15K applies
three of them — retrieval noise, visual ambiguity, and cross-modal conflict — at
**controlled dosages**, with an internal attention audit and a human ceiling in
the same instrument.

**Who created it and who funded it?**
GIB-15K maintainers.

## Composition

**What do the instances represent?**
Image–question–answer triples drawn from COCO 2017 and Visual Genome, each
labelled by hallucination sub-type (object / attribute / relation), and each
accompanied by a synthetic "retrieved passage". Every clean pair generates
adversarial variants along three axes.

**How many instances are there?**
* clean pairs: 15000
* variant rows (full): 325000
* variant rows (eval split): 59648
* conflict items: 15000
* of which counterfactually twinned: 4236

By axis: {"clean": 15000, "combined": 25000, "conflict": 15000, "noise": 75000, "visual": 195000}

By sub-type: {"object": 5000, "attribute": 5000, "relation": 5000}

**Is any information missing?**
Degraded images are *regenerable rather than redistributed* — see Distribution.
Some conflict items lack a counterfactual twin because no corpus partner with the
matching visual answer exists; those rows carry an empty `twin_id` and are
excluded from symmetry analyses rather than silently treated as twinned.

**Does the dataset contain sensitive or offensive content?**
No new imagery is created. The dataset inherits COCO and Visual Genome, which are
everyday photographs. Some contain identifiable people, as in the source
datasets. No new personal information is added, and no faces are annotated.

**Are there errors or noise?**
By design, yes: `answer_textual` is a deliberately injected falsehood, and the
degraded images are deliberately hard. These are the stimulus, not defects.
Unintentional error is bounded by round-trip validation (every question is
re-answered from the scene graph alone and discarded on mismatch) and by a 5%
stratified human spot-check whose acceptance rate is reported in the paper.

## Collection process

Questions are generated template-first from scene graphs (verifiable by
construction), with a minority of LLM paraphrases that are re-validated by the
same round-trip check. Hard negatives combine embedding-space nearest neighbours
with surgical single-fact mutations of the ground-truth passage. Entity swaps use
the WikiData sibling graph so substitutions are same-category and plausible.
Visual degradations are OpenCV transforms keyed to the answer object's bounding
box. All randomness is derived from content hashes, so the build is deterministic
under a fixed seed.

## Preprocessing / labelling

Sub-type labels come from the annotation source, not from post-hoc judgement:
COCO instances give object items, Visual Genome attributes and relationships give
the other two. The failure taxonomy is assigned automatically from two judge
verdicts, which is why taxonomy proportions can be reported over the full corpus.

## Uses

**What tasks is it suitable for?**
Diagnosing robustness of multimodal RAG; measuring modality preference under
conflict; counterfactual probing; calibration under distribution shift.

**What should it NOT be used for?**
It is a *diagnostic*, not a training set. Training on GIB-15K would teach a model
the specific mutation patterns rather than genuine grounding, and would invalidate
the benchmark for that model. It is also not a general-purpose VQA benchmark: its
clean split is easy by design, because it exists as the denominator for the
robustness ratios.

## Distribution

Released under **CC-BY 4.0** on GitHub and the HuggingFace Hub, tagged `v1.0`.

COCO and Visual Genome carry their own licences. GIB-15K therefore redistributes
**annotations plus transformation scripts**; degraded images are regenerated from
source by `make dataset`. Content hashes (`image_sha256`) in the schema let any
user verify their regenerated images are byte-identical to the ones evaluated.

## Maintenance

Versioned releases; the schema carries `schema_version`, and every result row
carries the dataset hash it was produced against, so a results file can always be
matched to the exact build that produced it.

## Ethics

The human study (Layer 4) is a non-sensitive perception task. Participants give
informed consent, are identified only by a random code, and no identifying
information is recorded. Participation is voluntary and may be ended at any time.


## Measured dataset characterisation

- Variant rows: **59,648**
- Sub-type balance (normalised entropy, 1.0 = perfectly even): **1.0** — {'attribute': 19721, 'object': 19986, 'relation': 19941}
- Distinct visual answers: **1447**, answer entropy **0.6374**, most-common-answer share **11.9%**
- Hard-negative difficulty (token Jaccard between truth and hardest distractor; higher = harder): mean **0.8887**, median **0.9**, p90 **1.0** (n=5,100)
- Conflict plausibility: **78.9%** of swaps stay in-class/family; 0 degenerate (no conflict)
- Counterfactual twins: **274** complete pairs (**42.5%** of conflict items twinned)
- Source provenance: {'coco': 11301, 'visual_genome': 48347}
