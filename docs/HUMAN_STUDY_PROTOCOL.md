# Layer 4 — human study protocol

Deliberately small and clean. The purpose is a **ceiling**, not a dataset: enough
data to establish where humans sit on each stressor, scored identically to the
models so the gap is legitimate.

---

## Design

| Parameter | Value | Rationale |
|---|---|---|
| Items | ~300 | Enough for per-axis cells of n ≈ 60 at 5 axes |
| Participants | 5–10 | Each sees ~60 % of items, giving overlap for agreement statistics |
| Stratification | axis × dose × sub_type, **budgeted per axis first** | Equal-per-cell would hand ~57 % of the study to the combined sub-grid (75 cells) and leave conflict (3 cells) with ~25 items |
| Order | Randomised per participant, seeded by participant id | Order effects do not align across participants |
| Backtracking | Disabled | Preserves first-pass judgement |
| Recorded | Answer + response time per item | The brief requires accuracy **and** time |
| Identity | Random 12-char code | No name, email, or IP is collected |

## Byte-identity with the model condition

Participants must see **exactly** what the models saw — the same degraded image
files and the same retrieved passages, in the same order. This is enforced, not
assumed:

```bash
make human-sample     # runs verify_identical(): re-hashes every served image
                      # against image_sha256 from the dataset
```

A mismatch is reported loudly. If images and hashes ever diverge, rebuild the
dataset before running the study — a gap measured against different stimuli is
not a finding.

## Scoring

Human answers go through the **identical cascade** used for models — normalised
exact match → alias match → the same pinned judge with the same frozen rubric.
Scoring humans generously by hand while scoring models strictly by code would
manufacture the very gap the study is trying to measure.

The same taxonomy classifier also runs on human answers, which yields a genuinely
interesting comparison: humans do occasionally show Knowledge Inertia, just far
less often than models.

## Reported statistics

- Accuracy per axis and per axis × dose
- **Krippendorff's α** — chosen over Fleiss' κ because participants see
  overlapping subsets, so the rating matrix has missing cells by construction
- Median response time overall and per axis
- Failure-mode proportions, directly comparable to the model tables

## Expected shape of the result

The target framing — humans ~90–95 % where GPT-4o falls to 40–60 % — is plausible
**specifically on the conflict axis**, because a person trivially trusts their
eyes over a contradicting paragraph.

It is *not* plausible everywhere, and the protocol is built to show that:

- **Conflict** — the gap should be at its widest.
- **Retrieval noise** — humans barely care how many distractor passages there
  are; they can just look. Gap wide.
- **Extreme blur / 5 % crop** — human perception genuinely fails too. **The gap
  should narrow here, and reporting that is what makes the study credible.**

`gap_by_stressor()` returns every cell both conditions cover, sorted by gap
descending. The narrowing cells stay in the table.

## Ethics

A non-sensitive perception task, so an informal ethics note in the paper
suffices — but it is written, and the consent screen is implemented in
`gib/human/app.py`:

- purpose, duration, and the presence of deliberately degraded/incorrect stimuli
  disclosed up front;
- explicitly voluntary, with withdrawal by closing the page;
- anonymous by construction — only a random code is stored;
- 18+ confirmation.

## Running it

```bash
make human-sample     # build + verify the sample
make human-app        # streamlit, share the URL with participants
make human-score      # score, summarise, write human_summary.json
```

`human_summary.json` is picked up automatically by `make figures` for
`fig11_human_gap` and `table7_human_gap`.

### Before recruiting

`make human-simulate` writes **simulated** responses so the scoring path,
agreement statistics, and gap figure can be verified end to end first. They land
in a separate file and require an explicit `--allow-simulated` to be scored — a
real run can never silently pick them up.
