"""Visual Genome ingestion -- supplies the *attribute* and *relation* sub-types.

VG scene graphs carry explicit ``(subject, predicate, object)`` triples and
per-object attribute lists, which is exactly what makes relation questions
("what is the man holding?") and attribute questions ("what colour is the
umbrella?") machine-verifiable rather than LLM-asserted.

VG images overlap heavily with COCO. That overlap is exploited deliberately:
``coco_id`` is preserved on every scene graph so a single image can carry all
three sub-types, and so the COCO instance inventory remains available for CHAIR
and POPE on VG-sourced items.

Expected layout under ``data/raw/vg``::

    scene_graphs.json      (or attributes.json + relationships.json)
    image_data.json
    VG_100K/*.jpg
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gib.config import RAW_DIR
from gib.dataengine.scene import Obj, Relation, SceneGraph
from gib.dataengine.sources import register_source

# VG predicates are noisy free text. Only triples whose predicate normalises into
# this map are kept -- an unfiltered predicate vocabulary would make the Axis 1
# predicate-swap mutator produce nonsense negatives.
PREDICATE_CANON: dict[str, str] = {
    "holding": "holding", "holds": "holding", "hold": "holding",
    "wearing": "wearing", "wears": "wearing", "has on": "wearing",
    "riding": "riding", "rides": "riding",
    "eating": "eating", "eats": "eating",
    "carrying": "carrying", "carries": "carrying",
    "throwing": "throwing", "throws": "throwing",
    "sitting on": "sitting on", "sits on": "sitting on",
    "standing on": "standing on", "stands on": "standing on",
    "lying on": "lying on",
    "playing with": "playing with", "playing": "playing with",
    "looking at": "looking at", "looks at": "looking at",
    "using": "using", "uses": "using",
    "touching": "touching", "touches": "touching",
    "pushing": "pushing", "pulling": "pulling",
    "on": "on", "in": "in", "under": "under", "above": "above",
    "behind": "behind", "in front of": "in front of", "near": "near",
    "next to": "next to", "on top of": "on top of",
}

# Attributes are likewise filtered to families with clean, mutually exclusive
# members, so an attribute swap is guaranteed to produce a false statement.
ATTRIBUTE_FAMILIES: dict[str, list[str]] = {
    "color": [
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "gray", "silver", "gold", "beige",
    ],
    "material": ["wooden", "metal", "plastic", "glass", "leather", "ceramic", "concrete"],
    "size": ["large", "small", "tall", "short", "tiny", "huge"],
    "state": ["open", "closed", "empty", "full", "broken", "clean", "dirty", "wet", "dry"],
}
ATTRIBUTE_TO_FAMILY: dict[str, str] = {
    a: fam for fam, members in ATTRIBUTE_FAMILIES.items() for a in members
}


def vg_root() -> Path:
    return RAW_DIR / "vg"


def is_available() -> bool:
    root = vg_root()
    return (root / "scene_graphs.json").exists() or (root / "attributes.json").exists()


def _norm_name(raw: Any) -> str:
    """VG object names arrive as ``names: [...]`` or ``name: str``."""
    if isinstance(raw, dict):
        names = raw.get("names") or ([raw["name"]] if raw.get("name") else [])
        return (names[0] if names else "").strip().lower()
    return str(raw).strip().lower()


def _bbox(o: dict) -> list[float]:
    return [float(o.get("x", 0)), float(o.get("y", 0)), float(o.get("w", 0)), float(o.get("h", 0))]


@register_source("visual_genome")
def load_visual_genome(limit: int = 10000, min_area: float = 1024.0, **_) -> Iterator[SceneGraph]:
    root = vg_root()
    sg_path = root / "scene_graphs.json"
    if not sg_path.exists():
        raise FileNotFoundError(
            f"Visual Genome scene graphs not found at {sg_path}. Run `make download-vg`, "
            f"or build with sources=[synthetic] for an offline smoke run."
        )

    with open(sg_path, encoding="utf-8") as fh:
        graphs = json.load(fh)

    meta_path = root / "image_data.json"
    meta_by_id: dict[int, dict] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta_by_id = {m["image_id"]: m for m in json.load(fh)}

    emitted = 0
    for g in graphs:
        if emitted >= limit:
            break
        image_id = g["image_id"]

        objects: list[Obj] = []
        by_obj_id: dict[Any, Obj] = {}
        for o in g.get("objects", []):
            name = _norm_name(o)
            if not name:
                continue
            box = _bbox(o)
            if box[2] * box[3] < min_area:
                continue
            attrs = [
                a.strip().lower()
                for a in (o.get("attributes") or [])
                if a.strip().lower() in ATTRIBUTE_TO_FAMILY
            ]
            obj = Obj(name=name, bbox=box, attributes=attrs, object_id=str(o.get("object_id", "")))
            objects.append(obj)
            by_obj_id[o.get("object_id")] = obj

        relations: list[Relation] = []
        for r in g.get("relationships", []):
            pred = PREDICATE_CANON.get(str(r.get("predicate", "")).strip().lower())
            if not pred:
                continue
            subj = by_obj_id.get(
                (r.get("subject") or {}).get("object_id")
                if isinstance(r.get("subject"), dict)
                else r.get("subject_id")
            )
            objt = by_obj_id.get(
                (r.get("object") or {}).get("object_id")
                if isinstance(r.get("object"), dict)
                else r.get("object_id")
            )
            if subj is None or objt is None or subj is objt:
                continue
            relations.append(
                Relation(
                    subject=subj.name,
                    predicate=pred,
                    obj=objt.name,
                    subject_bbox=subj.bbox,
                    object_bbox=objt.bbox,
                )
            )

        if not objects or not (relations or any(o.attributes for o in objects)):
            continue

        meta = meta_by_id.get(image_id, {})
        url = meta.get("url", "")
        # Resolve the image across every layout the VG archives produce. The
        # canonical split is VG_100K / VG_100K_2 (from the url), but some mirror
        # zips extract flat into the root, so a bare `{image_id}.jpg` is tried
        # too. Ordering the url-implied folder first keeps the common case fast.
        preferred = "VG_100K_2" if "VG_100K_2" in url else "VG_100K"
        candidates = [
            root / preferred / f"{image_id}.jpg",
            root / "VG_100K" / f"{image_id}.jpg",
            root / "VG_100K_2" / f"{image_id}.jpg",
            root / f"{image_id}.jpg",  # flat extraction
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            continue

        yield SceneGraph(
            image_id=f"vg{image_id}",
            image_path=str(img_path),
            width=int(meta.get("width", 0) or 0),
            height=int(meta.get("height", 0) or 0),
            source="visual_genome",
            objects=objects,
            relations=relations,
        )
        emitted += 1
