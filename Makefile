# GIB-15K — top-level pipeline targets.
#
# The entire figure set for the paper regenerates from `make figures`, and every
# layer below it is independently re-runnable without invalidating the others.

PY      ?= python
MODEL   ?= sim-open-7b
PAIRS   ?=
SOURCES ?=
DATASET ?= gib15k_eval.parquet
VERSION ?= v1.0

OPEN_MODELS   ?= llava-1.6-7b llava-1.6-13b qwen2.5-vl-7b
CLOSED_MODELS ?= gpt-4o gemini-1.5-pro
SIM_MODELS    ?= sim-frontier-a sim-frontier-b sim-open-13b sim-open-7b sim-small-open

.PHONY: help install install-heavy dataset dataset-demo validate characterize \
        croissant models eval eval-open eval-closed eval-sim probes interp \
        human-sample human-app human-score human-simulate figures paper-audit \
        render-paper build-paper dev-gate release test lint clean clean-results docker smoke

help:
	@echo "GIB-15K pipeline"
	@echo ""
	@echo "  make install         core deps (CPU: data engine, analysis, figures)"
	@echo "  make install-heavy   + torch/transformers/faiss (open-model eval)"
	@echo ""
	@echo "  make dataset         Layer 1: build GIB-15K from COCO + Visual Genome"
	@echo "  make dataset-demo    Layer 1: offline synthetic build (no downloads)"
	@echo "  make validate        check the built dataset against its contract"
	@echo ""
	@echo "  make eval MODEL=…    Layer 2: run one model  (make models to list)"
	@echo "  make eval-open       Layer 2: the open-source tier"
	@echo "  make eval-closed     Layer 2: the API tier (stratified subset)"
	@echo ""
	@echo "  make interp MODEL=…  Layer 3: attention probe (Track A only)"
	@echo "  make human-sample    Layer 4: build the study sample"
	@echo "  make human-app       Layer 4: serve the collection UI"
	@echo "  make human-score     Layer 4: score collected responses"
	@echo ""
	@echo "  make figures         Layer 5: regenerate EVERY figure and table"
	@echo "  make release         Layer 5: datasheet + Croissant + HF card"
	@echo ""
	@echo "  make smoke           full offline end-to-end run (~5 min)"
	@echo "  make test            unit tests"

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

install:
	$(PY) -m pip install -r requirements.txt

install-heavy:
	$(PY) -m pip install -r requirements.txt -r requirements-heavy.txt

# --------------------------------------------------------------------------- #
# Layer 1
# --------------------------------------------------------------------------- #

dataset:
	$(PY) -m gib.cli dataset $(if $(PAIRS),--pairs $(PAIRS)) $(if $(SOURCES),--sources $(SOURCES))

dataset-demo:
	$(PY) -m gib.cli dataset --pairs 1200 --sources synthetic

validate:
	$(PY) -m gib.cli validate --dataset $(DATASET)

models:
	$(PY) -m gib.cli models

# --------------------------------------------------------------------------- #
# Layer 2
# --------------------------------------------------------------------------- #

eval:
	$(PY) -m gib.cli eval --model $(MODEL) --dataset $(DATASET)

# Open models: the full eval split — local GPUs are amortised.
eval-open:
	@for m in $(OPEN_MODELS); do \
		echo "=== $$m ==="; \
		$(PY) -m gib.cli eval --model $$m --dataset $(DATASET) || exit 1; \
	done

# Closed models: a stratified subset, N=10 samples. The response cache makes the
# bill a one-time cost; a re-run is free.
eval-closed:
	@for m in $(CLOSED_MODELS); do \
		echo "=== $$m ==="; \
		$(PY) -m gib.cli eval --model $$m --dataset $(DATASET) \
			--limit 5000 --n-samples 10 --proxy-model llava-1.6-7b || exit 1; \
	done

eval-sim:
	@for m in $(SIM_MODELS); do \
		echo "=== $$m ==="; \
		$(PY) -m gib.cli eval --model $$m --offline-judge --n-samples 8 || exit 1; \
		$(PY) -m gib.cli probes --model $$m || exit 1; \
	done

# CHAIR (captioning mode) + POPE (yes/no probing). Separate interaction modes
# from the main QA loop, so they run as their own pass.
probes:
	$(PY) -m gib.cli probes --model $(MODEL) --dataset $(DATASET)

# --------------------------------------------------------------------------- #
# Layers 3 & 4
# --------------------------------------------------------------------------- #

interp:
	$(PY) -m gib.cli interp --model $(MODEL) --dataset $(DATASET)

human-sample:
	$(PY) -m gib.cli human-sample --dataset $(DATASET)

human-app:
	streamlit run gib/human/app.py

human-score:
	$(PY) -m gib.cli human-score

# SIMULATED participants — exercises Layer 4's scoring path before recruitment.
human-simulate:
	$(PY) -m gib.cli human-simulate

# --------------------------------------------------------------------------- #
# Layer 5
# --------------------------------------------------------------------------- #

figures:
	$(PY) -m gib.cli figures

release: paper-audit
	$(PY) -m gib.cli release --version $(VERSION)

# The paper-audit gate: every number in the manuscript must match provenance
# before a public release. Depends on figures having been generated.
paper-audit:
	$(PY) -m gib.cli paper-audit

# Render the manuscript template, filling {{provenance.key}} placeholders from
# measured numbers so no quoted value can go stale.
render-paper:
	$(PY) -m gib.cli render-paper --template paper/gib15k.md.tmpl --out paper/gib15k.md

# Full paper build: render from provenance, audit, then (if pandoc is present)
# emit a PDF. Falls back to Markdown-only when no TeX/pandoc is installed.
build-paper: render-paper paper-audit
	@command -v pandoc >/dev/null 2>&1 && \
		pandoc paper/gib15k.md -o paper/gib15k.pdf --pdf-engine=xelatex \
			-V geometry:margin=1in -V fontsize=11pt && echo "wrote paper/gib15k.pdf" || \
		echo "pandoc not found — paper/gib15k.md is ready; see paper/README.md for render commands"

# GIB-1K-dev-first safeguard: the entire pipeline must pass end-to-end on a small
# subset before committing full-scale compute. Catches broken architecture early.
dev-gate:
	$(PY) -m gib.cli dataset --pairs 1000 --sources synthetic
	$(PY) -m gib.cli validate
	$(PY) -m gib.cli characterize
	$(PY) -m gib.cli croissant
	$(PY) -m gib.cli eval --model sim-open-7b --offline-judge --n-samples 4
	$(PY) -m gib.cli probes --model sim-open-7b
	$(PY) -m gib.cli figures
	@echo "dev-gate PASSED — pipeline is sound end-to-end; safe to scale."

# --------------------------------------------------------------------------- #
# Dev
# --------------------------------------------------------------------------- #

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check gib/ tests/ || true

# Recipes invoke the CLI directly rather than recursing through $(MAKE): on
# Windows, MAKE expands to a path containing spaces and parentheses
# ("C:/Program Files (x86)/...") which the shell cannot parse.
smoke: dataset-demo validate eval-sim
	$(PY) -m gib.cli interp --model sim-open-7b --n-conflict 120
	$(PY) -m gib.cli human-sample
	$(PY) -m gib.cli human-simulate
	$(PY) -m gib.cli human-score --allow-simulated
	$(PY) -m gib.cli figures
	$(PY) -m gib.cli release --version $(VERSION)
	@echo ""
	@echo "Smoke run complete - all five layers exercised."
	@echo "Figures in ./figures, watermarked SIMULATED. Not results."

docker:
	docker build -t gib15k:$(VERSION) .

clean-results:
	rm -rf results figures .cache

clean: clean-results
	rm -rf data/gib15k data/images data/release
