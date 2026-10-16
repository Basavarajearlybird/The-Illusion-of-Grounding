"""Versioned data contract for GIB-15K.

Every layer of the system reads and writes this one flat schema. Keeping it flat
is what lets the evaluation engine stay axis-agnostic: a runner never asks "which
axis am I on?", it just consumes `image_path`, `question` and `retrieved_docs`.

Two fields carry most of the intellectual weight:

* ``answer_visual``  -- the truth as shown in the image.
* ``answer_textual`` -- the falsehood injected into the retrieved passages.

Because both live in the *data*, the failure taxonomy (Layer 2.4) is a
deterministic function of a judge verdict rather than a hand-labelled annotation,
which is what makes taxonomy proportions reportable across the whole corpus.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


class SubType(str, Enum):
    """Hallucination sub-type, per project requirement 2."""

    OBJECT = "object"
    ATTRIBUTE = "attribute"
    RELATION = "relation"


class Axis(str, Enum):
    """Stressor axis. ``CLEAN`` is the unperturbed control condition."""

    CLEAN = "clean"
    NOISE = "noise"
    VISUAL = "visual"
    CONFLICT = "conflict"
    COMBINED = "combined"  # the visual x noise sub-grid behind the heatmap


class Degradation(str, Enum):
    """Visual degradation family (Axis 2)."""

    NONE = "none"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    CROP = "crop"


class FailureMode(str, Enum):
    """The formal failure taxonomy (paper requirement B)."""

    CORRECT = "correct"
    KNOWLEDGE_INERTIA = "knowledge_inertia"
    SENSORY_CAPTURE = "sensory_capture"
    REASONING_COLLAPSE = "reasoning_collapse"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class CleanPair:
    """A clean image-QA pair as produced by Layer 1.1 (source ingestion).

    ``ground_truth_doc`` is generated at ingestion time rather than later,
    because every retrieval-side perturbation is defined as a mutation of it.
    """

    pair_id: str
    source: str  # coco | visual_genome | synthetic
    image_id: str
    image_path: str
    image_sha256: str
    question: str
    answer: str
    sub_type: SubType
    bbox: list[float]  # [x, y, w, h] of the answer-bearing object
    ground_truth_doc: str
    # Scene-graph provenance, kept so the mutators can operate structurally
    # rather than by string surgery.
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    attribute: str = ""
    qa_origin: str = "template"  # template | llm_paraphrase
    validated: bool = True

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["sub_type"] = self.sub_type.value
        return row


@dataclass
class Variant:
    """One evaluable row: a clean pair viewed through one stressor at one dose."""

    pair_id: str
    variant_id: str
    sub_type: SubType
    axis: Axis
    dose: str  # sigma / kernel px / crop % / distractor count, or "n/a"

    image_path: str
    image_sha256: str

    question: str
    answer_visual: str
    answer_textual: str  # "" when no falsehood was injected

    retrieved_docs: list[str] = field(default_factory=list)
    doc_positions: list[int] = field(default_factory=list)
    gt_doc_position: int = -1

    twin_id: str = ""  # links counterfactual pairs
    bbox: list[float] = field(default_factory=list)

    # Provenance for reproducibility / free ablations
    degradation: Degradation = Degradation.NONE
    degradation_param: float = 0.0
    motion_angle: float = 0.0
    n_distractors: int = 0
    source: str = ""
    seed: int = 0
    schema_version: str = SCHEMA_VERSION

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["sub_type"] = self.sub_type.value
        row["axis"] = self.axis.value
        row["degradation"] = self.degradation.value
        return row


def make_variant_id(pair_id: str, axis: Axis | str, dose: str, suffix: str = "") -> str:
    """`{pair_id}-{axis}-{dose}` per the architecture, with an optional suffix
    used to disambiguate counterfactual twins and degradation families."""
    axis_v = axis.value if isinstance(axis, Axis) else axis
    dose_slug = str(dose).replace(".", "p").replace("/", "-").replace(" ", "")
    parts = [pair_id, axis_v, dose_slug]
    if suffix:
        parts.append(suffix)
    return "-".join(parts)


# --------------------------------------------------------------------------- #
# Arrow schema
# --------------------------------------------------------------------------- #

VARIANT_COLUMNS: Sequence[str] = (
    "pair_id",
    "variant_id",
    "sub_type",
    "axis",
    "dose",
    "image_path",
    "image_sha256",
    "question",
    "answer_visual",
    "answer_textual",
    "retrieved_docs",
    "doc_positions",
    "gt_doc_position",
    "twin_id",
    "bbox",
    "degradation",
    "degradation_param",
    "motion_angle",
    "n_distractors",
    "source",
    "seed",
    "schema_version",
)

RESULT_COLUMNS: Sequence[str] = (
    "run_id",
    "model",
    "track",  # A (white-box) | B (black-box)
    "variant_id",
    "pair_id",
    "axis",
    "dose",
    "sub_type",
    "answer",  # the model's generated answer
    "is_correct",
    "matches_textual",
    "failure_mode",
    "confidence",
    "trust_index",
    "csr",  # REFIND context-sensitivity ratio
    "latency_s",
    "n_samples",
    "judge_verdict",
    "git_sha",
    "dataset_hash",
    "adapter_version",
)


def arrow_schema_variants():
    """Built lazily so the schema module stays importable without pyarrow."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("pair_id", pa.string()),
            pa.field("variant_id", pa.string()),
            pa.field("sub_type", pa.string()),
            pa.field("axis", pa.string()),
            pa.field("dose", pa.string()),
            pa.field("image_path", pa.string()),
            pa.field("image_sha256", pa.string()),
            pa.field("question", pa.string()),
            pa.field("answer_visual", pa.string()),
            pa.field("answer_textual", pa.string()),
            pa.field("retrieved_docs", pa.list_(pa.string())),
            pa.field("doc_positions", pa.list_(pa.int32())),
            pa.field("gt_doc_position", pa.int32()),
            pa.field("twin_id", pa.string()),
            pa.field("bbox", pa.list_(pa.float64())),
            pa.field("degradation", pa.string()),
            pa.field("degradation_param", pa.float64()),
            pa.field("motion_angle", pa.float64()),
            pa.field("n_distractors", pa.int32()),
            pa.field("source", pa.string()),
            pa.field("seed", pa.int64()),
            pa.field("schema_version", pa.string()),
        ]
    )


# --------------------------------------------------------------------------- #
# Content addressing
# --------------------------------------------------------------------------- #


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Order-independent hash of a JSON-able object; used for dataset versioning
    and for the response-cache key."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dataset_hash(rows: Iterable[dict[str, Any]]) -> str:
    """Hash of the variant_id set: cheap, stable, and enough to detect that an
    eval was run against a different dataset build."""
    ids = sorted(r["variant_id"] for r in rows)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]
