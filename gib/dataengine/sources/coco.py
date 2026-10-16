"""COCO 2017 ingestion -- supplies the *object* sub-type.

Instance segmentations give a ground-truth object inventory per image, and the
bounding boxes are what Axis 2's low-resource crop is keyed to. The architecture
calls for ingesting ``instances_train2017.json`` in full, so this reads the raw
annotation JSON directly rather than depending on pycocotools.

Expected layout under ``data/raw/coco``::

    annotations/instances_train2017.json
    annotations/instances_val2017.json
    train2017/*.jpg
    val2017/*.jpg
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from gib.config import RAW_DIR
from gib.dataengine.scene import Obj, Relation, SceneGraph
from gib.dataengine.sources import register_source


def _spatial_predicate(a: list[float], b: list[float]) -> str:
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    if abs(ax - bx) >= abs(ay - by):
        return "to the left of" if ax < bx else "to the right of"
    return "above" if ay < by else "below"


def coco_root() -> Path:
    return RAW_DIR / "coco"


def split_ready(split: str) -> bool:
    """A split is usable only if BOTH its annotations and its images are on disk.

    Checking the annotation JSON alone is not enough: COCO's
    ``annotations_trainval2017.zip`` ships the train *and* val instance files
    together, so ``instances_train2017.json`` can be present while the 18 GB
    train image set is still downloading. Selecting that split would then skip
    every image (none exist yet) and yield an empty corpus -- exactly the failure
    that slipped past the annotation-only check.
    """
    root = coco_root()
    ann = root / "annotations" / f"instances_{split}.json"
    img_dir = root / split
    return ann.exists() and img_dir.is_dir() and any(img_dir.glob("*.jpg"))


def is_available(split: str = "train2017") -> bool:
    return split_ready(split)


def resolve_split(split: str | None) -> str:
    """Pick a *ready* split, preferring the requested one.

    ``split=None``/"auto" means "whatever is fully on disk", trying train2017
    first. This lets a dev build run against a val-only download while the train
    image set is still coming down.
    """
    if split and split not in ("auto", None) and split_ready(split):
        return split
    for cand in ("train2017", "val2017"):
        if split_ready(cand):
            return cand
    # Nothing is fully ready; return the request (or default) so the caller
    # raises a clear FileNotFoundError naming the missing path.
    return split if (split and split != "auto") else "train2017"


@register_source("coco")
def load_coco(
    limit: int = 5000,
    split: str = "auto",
    min_area: float = 1024.0,
    **_,
) -> Iterator[SceneGraph]:
    """Yield scene graphs from COCO instance annotations.

    Objects smaller than ``min_area`` px^2 are dropped: the crop degradation
    reduces the answer object to 5% of its visible area, and a tiny source object
    would be sub-pixel at the extreme dose, confounding "the model cannot see it"
    with "there is nothing to see".
    """
    root = coco_root()
    split = resolve_split(split)
    ann_path = root / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_path}. Run `bash scripts/download_coco.sh`, "
            f"or build with sources=[synthetic] for an offline smoke run."
        )

    with open(ann_path, encoding="utf-8") as fh:
        data = json.load(fh)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}

    per_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann.get("area", 0.0) < min_area:
            continue
        per_image[ann["image_id"]].append(ann)

    emitted = 0
    for image_id in sorted(per_image):
        if emitted >= limit:
            break
        anns = per_image[image_id]
        if len(anns) < 2:  # need a scene, not a single object
            continue
        meta = images.get(image_id)
        if meta is None:
            continue
        img_path = root / split / meta["file_name"]
        if not img_path.exists():
            continue

        objects = [
            Obj(
                name=cats[a["category_id"]],
                bbox=[float(v) for v in a["bbox"]],
                attributes=[],  # COCO carries no attributes; VG supplies those
                object_id=str(a["id"]),
            )
            for a in anns
        ]
        relations = [
            Relation(
                subject=a.name,
                predicate=_spatial_predicate(a.bbox, b.bbox),
                obj=b.name,
                subject_bbox=a.bbox,
                object_bbox=b.bbox,
            )
            for a, b in zip(objects, objects[1:])
        ]

        yield SceneGraph(
            image_id=f"coco{image_id}",
            image_path=str(img_path),
            width=int(meta["width"]),
            height=int(meta["height"]),
            source="coco",
            objects=objects,
            relations=relations,
        )
        emitted += 1


def load_instance_index(split: str | None = None) -> dict[str, set[str]]:
    """image_id -> set of present category names.

    This is the ground-truth inventory the CHAIR metric scores captions against
    and the positive/negative pool the POPE probe generator samples from.

    ``split=None`` merges every split present on disk. Merging is safe because
    COCO image ids are globally unique, and it means CHAIR/POPE find the
    inventory whether the corpus was built from train, val, or both -- otherwise
    a val-built dev corpus would silently get an empty inventory and every
    caption would score as fully hallucinated.
    """
    root = coco_root()
    splits = [split] if split else ["train2017", "val2017"]
    index: dict[str, set[str]] = defaultdict(set)
    for sp in splits:
        ann_path = root / "annotations" / f"instances_{sp}.json"
        if not ann_path.exists():
            continue
        with open(ann_path, encoding="utf-8") as fh:
            data = json.load(fh)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        for ann in data["annotations"]:
            index[f"coco{ann['image_id']}"].add(cats[ann["category_id"]])
    return dict(index)


COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
