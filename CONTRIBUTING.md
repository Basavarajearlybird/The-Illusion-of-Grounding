# Contributing to GIB-15K

Thanks for your interest. GIB-15K is a *diagnostic benchmark*, so contributions
are held to a reproducibility bar first and a code-style bar second.

## The one rule that matters

**Every claim must be reproducible from a command.** If you add a metric, a
figure, or a number that appears anywhere in the paper or README, it must be
emitted by `make figures` (or another `make` target) and land in
`figures/paper_numbers.json`. Numbers typed by hand are not accepted — the
`make paper-audit` gate exists to catch them.

## Before you open a PR

```bash
make install            # core deps (CPU only)
make test               # unit + integration tests must pass
make lint               # ruff, no new warnings
make smoke              # full offline pipeline must run end-to-end
```

CI runs all four on Python 3.10–3.12. A PR that breaks the offline smoke will not
be merged, because the smoke is the guarantee that the pipeline still works for a
reviewer with no GPU and no API keys.

## Adding to the benchmark

| You want to add… | Do this |
|---|---|
| A model | one adapter file under `gib/adapters/`, registered via `@register_adapter`. Track A instrumentation (Trust Index, CSR, VAM) comes free. |
| A metric | drop it in `gib/eval/metrics/`, surface it as a column in `runner.evaluate_row` or as a pass in `gib/eval/probes.py`. Add a unit test. |
| A stressor axis | add to `schema.Axis`, write a generator emitting `Variant` rows, call it from `dataengine/build.build_variants`. |
| A source | normalise into `dataengine/scene.SceneGraph` and register under `dataengine/sources/`. |

Every new module needs a test in `tests/`, and every scientific decision needs a
one-line comment explaining *why* (see the existing code — the comments justify
choices, they don't restate the code).

## Determinism is non-negotiable

New randomness must derive from `gib.seeds.derive_seed(...)` on stable
identifiers, never from a global RNG. A partial re-run must produce byte-identical
output to a full run; there is a test that enforces this.

## Never do

- Put a simulated (`sim-*`) number into the paper. They are watermarked for a
  reason.
- Let a model judge its own outputs (`assert_no_circularity` will stop you).
- Ship degraded images — they are regenerable and verified by hash.

## Reporting issues

Use the templates under `.github/ISSUE_TEMPLATE/`. For a data issue, include the
`variant_id`; for a build issue, include `data/gib15k/variants/manifest.json`.
