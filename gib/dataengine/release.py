"""Release engineering (Layer 5) -- CC-BY on GitHub + HuggingFace Hub.

Emits the datasheet (Gebru et al. format), Croissant metadata, the HF dataset
card, and a licence note, then tags everything ``v1.0``.

**The licensing position, which the release docs must state plainly.** COCO and
Visual Genome carry their own licences. GIB-15K therefore redistributes
*annotations plus transformation scripts*, and the degraded images are
regenerable from source rather than shipped. That sidesteps most redistribution
questions: a user downloads COCO/VG under their original terms and runs
``make dataset`` to reproduce the exact bytes, verified by the content hashes
already in the schema.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from gib.config import RELEASE_DIR, VARIANTS_DIR

# The canonical Croissant 1.0 @context. The reference mlcroissant validator
# rejects a partial context (it expects every one of these keys), so the full
# standard block is embedded verbatim rather than trimmed.
CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


def validate_croissant(path: Path | str) -> dict:
    """Validate a Croissant JSON-LD file with the reference ``mlcroissant`` loader.

    A Croissant file that does not parse is worse than none — it signals machine
    readability the dataset does not actually have. This runs the official
    validator so the claim is verified, not asserted. Returns a report dict; when
    ``mlcroissant`` is not installed it degrades to a structural check rather than
    failing the release.
    """
    path = Path(path)
    report: dict[str, Any] = {"file": str(path), "exists": path.exists()}
    if not path.exists():
        report["ok"] = False
        report["error"] = "file not found"
        return report

    raw = json.loads(path.read_text(encoding="utf-8"))
    # Structural checks that do not need the library.
    report["has_context"] = "@context" in raw
    report["has_recordset"] = bool(raw.get("recordSet"))
    report["n_fields"] = sum(len(rs.get("field", [])) for rs in raw.get("recordSet", []))

    try:
        import mlcroissant as mlc

        issues = mlc.Dataset(jsonld=str(path)).metadata.issues
        errors = list(getattr(issues, "errors", []) or [])
        warnings = list(getattr(issues, "warnings", []) or [])
        report["validator"] = "mlcroissant"
        report["errors"] = [str(e) for e in errors]
        report["warnings"] = [str(w) for w in warnings]
        report["ok"] = len(errors) == 0
    except ImportError:
        report["validator"] = "structural-only (mlcroissant not installed)"
        report["ok"] = report["has_context"] and report["has_recordset"] and report["n_fields"] > 0
    except Exception as exc:  # noqa: BLE001
        report["validator"] = "mlcroissant"
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _manifest() -> dict:
    p = VARIANTS_DIR / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _semver(version: str) -> str:
    """Normalise a tag like ``v1.0`` / ``v0.1-dev`` to MAJOR.MINOR.PATCH.

    Croissant validation warns on non-semver versions; padding the numeric core
    to three components and dropping any pre-release suffix keeps the metadata
    warning-clean while preserving the tag the user passed elsewhere.
    """
    import re

    core = re.match(r"v?(\d+(?:\.\d+)*)", version.strip())
    parts = (core.group(1).split(".") if core else ["1", "0", "0"])
    parts = (parts + ["0", "0", "0"])[:3]
    return ".".join(parts)


def build_croissant(version: str = "v1.0") -> dict:
    """Emit a Croissant 1.0 metadata document that passes the reference validator.

    Validity requires more than a field list: ``conformsTo``, a ``distribution``
    of concrete file objects, and per-field ``source`` extractions pointing into
    that distribution. Each field is typed with a real schema.org DataType and
    (where it maps cleanly) a schema.org semantic property, so the record set is
    machine-interpretable rather than just self-described.
    """
    m = _manifest()
    dist_id = "gib15k-parquet"
    # (name, DataType, column, description, optional schema.org property)
    fields = [
        ("pair_id", "sc:Text", "pair_id", "Stable id shared across all variants of a pair.", None),
        ("variant_id", "sc:Text", "variant_id", "{pair_id}-{axis}-{dose}; unique per row.", None),
        ("sub_type", "sc:Text", "sub_type", "Sub-type: object / attribute / relation.", None),
        ("axis", "sc:Text", "axis", "Stressor axis: clean/noise/visual/conflict/combined.", None),
        ("dose", "sc:Text", "dose", "Sigma, kernel px, crop %, or distractor count.", None),
        ("image_path", "sc:Text", "image_path", "Path to the (possibly degraded) image.", None),
        ("image_sha256", "sc:Text", "image_sha256", "SHA-256 of the served image bytes.", None),
        ("question", "sc:Text", "question", "The visual question.", None),
        ("answer_visual", "sc:Text", "answer_visual", "Ground truth shown in the image.", None),
        ("answer_textual", "sc:Text", "answer_textual", "Injected falsehood; empty if none.", None),
        ("gt_doc_position", "sc:Integer", "gt_doc_position", "Index of the true passage; -1 if absent.", None),
        ("twin_id", "sc:Text", "twin_id", "Links counterfactual twin pairs.", None),
    ]

    def _field(name, dtype, column, desc, prop):
        f = {
            "@type": "cr:Field",
            "@id": f"variants/{name}",
            "name": name,
            "description": desc,
            "dataType": [dtype] + ([prop] if prop else []),
            "source": {
                "fileSet": {"@id": dist_id},
                "extract": {"column": column},
            },
        }
        return f

    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "GIB-15K",
        "description": (
            "A diagnostic benchmark for multimodal RAG under retrieval noise, visual "
            "ambiguity, and cross-modal conflict. 15,000 image-QA pairs labelled by "
            "hallucination sub-type, each with retrieval-noised, visually degraded, "
            "and cross-modal-conflicted adversarial variants at controlled dosages."
        ),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": "https://huggingface.co/datasets/gib15k",
        "version": _semver(version),
        "datePublished": date.today().isoformat(),
        "citeAs": "@misc{gib15k, title={The Illusion of Grounding}, year={2026}}",
        "keywords": [
            "multimodal", "retrieval-augmented generation", "hallucination",
            "vision-language models", "benchmark", "robustness",
        ],
        "distribution": [
            {
                "@type": "cr:FileSet",
                "@id": dist_id,
                "name": dist_id,
                "description": (
                    f"{m.get('n_variants_full', 'N')} adversarial variant rows, "
                    f"one parquet per axis."
                ),
                "encodingFormat": "application/vnd.apache.parquet",
                "includes": "variants/**/*.parquet",
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "variants",
                "name": "variants",
                "description": f"{m.get('n_variants_full', 'N')} adversarial variant rows.",
                "field": [_field(*f) for f in fields],
            }
        ],
    }


DATASHEET = """# Datasheet for GIB-15K

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
{maintainer}

## Composition

**What do the instances represent?**
Image–question–answer triples drawn from COCO 2017 and Visual Genome, each
labelled by hallucination sub-type (object / attribute / relation), and each
accompanied by a synthetic "retrieved passage". Every clean pair generates
adversarial variants along three axes.

**How many instances are there?**
{composition}

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

Released under **CC-BY 4.0** on GitHub and the HuggingFace Hub, tagged `{version}`.

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
"""

HF_CARD = """---
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
"""


def build_release(version: str = "v1.0", maintainer: str = "GIB-15K maintainers.") -> Path:
    out = Path(RELEASE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    m = _manifest()

    composition = "\n".join(
        f"* {k}: {v}"
        for k, v in [
            ("clean pairs", m.get("n_clean_pairs", "?")),
            ("variant rows (full)", m.get("n_variants_full", "?")),
            ("variant rows (eval split)", m.get("n_variants_eval", "?")),
            ("conflict items", m.get("conflict_items", "?")),
            ("of which counterfactually twinned", m.get("conflict_twinned", "?")),
        ]
    ) + "\n\nBy axis: " + json.dumps(m.get("by_axis", {})) + \
        "\n\nBy sub-type: " + json.dumps(m.get("by_sub_type", {}))

    (out / "DATASHEET.md").write_text(
        DATASHEET.format(maintainer=maintainer, composition=composition, version=version),
        encoding="utf-8",
    )
    (out / "README.md").write_text(HF_CARD, encoding="utf-8")
    (out / "croissant.json").write_text(
        json.dumps(build_croissant(version), indent=2), encoding="utf-8"
    )
    (out / "LICENSE").write_text(
        "GIB-15K annotations and code: CC-BY 4.0.\n\n"
        "Source imagery remains under its original licence:\n"
        "  * COCO 2017 — see https://cocodataset.org/#termsofuse\n"
        "  * Visual Genome — CC-BY 4.0, see https://homes.cs.washington.edu/~ranjay/visualgenome/\n\n"
        "Degraded images are NOT redistributed. They are regenerated from source by\n"
        "`make dataset` and verified byte-for-byte against the `image_sha256` column.\n",
        encoding="utf-8",
    )
    if (VARIANTS_DIR / "manifest.json").exists():
        shutil.copy(VARIANTS_DIR / "manifest.json", out / "build_manifest.json")

    # -- validate the Croissant file we just wrote --------------------------- #
    croissant_report = validate_croissant(out / "croissant.json")
    (out / "croissant_validation.json").write_text(
        json.dumps(croissant_report, indent=2), encoding="utf-8"
    )
    status = "PASS" if croissant_report.get("ok") else "FAIL"
    print(f"[release] croissant validation: {status} "
          f"({croissant_report.get('validator')}, {croissant_report.get('n_fields')} fields)")

    # -- append measured characterisation to the datasheet ------------------- #
    char_path = VARIANTS_DIR / "characterization.md"
    if char_path.exists():
        datasheet = out / "DATASHEET.md"
        datasheet.write_text(
            datasheet.read_text(encoding="utf-8") + "\n\n" + char_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print("[release] appended measured characterisation to datasheet")

    print(f"[release] {version} bundle -> {out}")
    for f in sorted(out.iterdir()):
        print(f"           {f.name}")
    return out
