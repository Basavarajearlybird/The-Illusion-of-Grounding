---
name: Bug report
about: A pipeline, build, or data problem
labels: bug
---

**Layer** (Data engine / Eval / Interp / Human / Analysis):

**What happened**

**Reproduce**
```bash
# the exact command(s)
```

**Provenance** (attach the relevant one)
- Data issue: the `variant_id`, and `data/gib15k/variants/manifest.json`
- Build issue: `manifest.json` + the failing `make` target
- Eval issue: `results/model=<name>/run_manifest.json`

**Environment**
- OS / Python:
- `pip freeze | grep -E 'numpy|pandas|torch|transformers|opencv'`:
