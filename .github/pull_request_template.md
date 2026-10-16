## What this changes

<!-- one or two sentences -->

## Reproducibility checklist

- [ ] `make test` passes
- [ ] `make lint` clean (no new ruff warnings)
- [ ] `make smoke` runs end-to-end offline
- [ ] Any new number that appears in a figure/table is emitted by `make figures`
      into `paper_numbers.json` (not typed by hand)
- [ ] New randomness derives from `gib.seeds.derive_seed`, not a global RNG
- [ ] New modules have tests

## If this touches the dataset schema or a metric

- [ ] `schema_version` bumped if the data contract changed
- [ ] `make paper-audit` still passes
- [ ] Rationale for the scientific decision is commented in the code
