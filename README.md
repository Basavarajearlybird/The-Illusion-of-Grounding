# Paper — GIB-15K registered experimental design

`gib15k.md.tmpl` is the manuscript **template**. Every numeric claim is a
provenance placeholder (double-brace key syntax) or an inline audit tag
(`value⟨key⟩`), never a hand-typed number.

## Build

```bash
make figures          # produces figures/paper_numbers.json + provenance.csv
make render-paper     # fills the template -> paper/gib15k.md
make paper-audit      # verifies every reference resolves + matches provenance
```

`make release` runs `paper-audit` as a dependency, so a release cannot ship a
manuscript whose numbers disagree with what the pipeline measured.

## Render to PDF / HTML

The rendered `gib15k.md` is standard Pandoc-flavoured Markdown with a YAML front
matter block:

```bash
# PDF (needs a TeX engine)
pandoc paper/gib15k.md -o paper/gib15k.pdf --pdf-engine=xelatex \
    -V geometry:margin=1in -V fontsize=11pt

# self-contained HTML (no TeX needed)
pandoc paper/gib15k.md -o paper/gib15k.html --standalone --mathjax
```

## Why a template instead of a written paper

*The paper can never still say "ADS = 0.87" if the rebuild measured something
else.* A number pulled from provenance at render time cannot go stale, and
`paper-audit` fails the build if a placeholder is unresolved or an inline value
drifts. The claim **verbs** are also generated from the statistics
(`gib.analysis.claims`): a strong effect renders "demonstrate", a wrong-signed or
null effect renders "do not support" — so the manuscript's language tracks the
evidence automatically.

Until a real model zoo has been run, model-facing numbers come from simulated
adapters and the figures are watermarked SIMULATED; the template says so in
Section 4.
