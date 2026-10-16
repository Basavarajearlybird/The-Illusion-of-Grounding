"""Synthetic scene source -- the offline / smoke-test corpus.

COCO and Visual Genome together are ~45 GB of downloads. That is the right input
for the real benchmark build, but it makes the pipeline impossible to exercise on
a laptop, in CI, or on a reviewer's machine. This source procedurally renders
scenes together with *exact* scene graphs, so every layer downstream -- QA
templating, bbox-keyed cropping, hard negatives, conflict twins, the eval engine,
the figures -- runs end to end with zero downloads.

It is a fixture, not a scientific artifact. `make dataset` on the real sources is
what produces the released GIB-15K; this is what proves the code that produces it
works. The distinction is stated in the datasheet.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageDraw

from gib.config import IMAGES_CLEAN_DIR
from gib.dataengine.scene import Obj, Relation, SceneGraph
from gib.dataengine.sources import register_source
from gib.seeds import rng_for

# Vocabulary chosen to overlap COCO categories so that WikiData sibling lookup
# and the POPE probe generator behave the same as on the real corpus.
CATEGORIES: dict[str, str] = {
    "apple": "circle",
    "orange": "circle",
    "banana": "arc",
    "umbrella": "triangle",
    "frisbee": "circle",
    "kite": "diamond",
    "backpack": "rect",
    "laptop": "rect",
    "book": "rect",
    "clock": "circle",
    "vase": "trapezoid",
    "chair": "rect",
    "bicycle": "wheels",
    "dog": "blob",
    "cat": "blob",
    "bird": "triangle",
    "cup": "trapezoid",
    "bottle": "bottle",
    "person": "person",
    "ball": "circle",
}

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (203, 60, 51),
    "green": (58, 145, 80),
    "blue": (52, 96, 176),
    "yellow": (222, 184, 47),
    "orange": (224, 128, 44),
    "purple": (126, 74, 168),
    "brown": (124, 88, 60),
    "black": (38, 38, 40),
    "white": (238, 238, 235),
    "gray": (135, 138, 142),
}

# Predicates a person can take with a held object -- these give the *relation*
# sub-type genuinely non-spatial triples, which is what VG contributes on the
# real corpus.
HELD_PREDICATES = ["holding", "throwing", "carrying", "using", "touching"]

CANVAS = (640, 480)
BACKGROUNDS = [(240, 240, 238), (226, 232, 236), (236, 232, 226), (232, 236, 230)]


def _draw_shape(
    draw: ImageDraw.ImageDraw, shape: str, box: tuple[float, float, float, float], color
) -> None:
    x, y, w, h = box
    x0, y0, x1, y1 = x, y, x + w, y + h
    outline = tuple(max(0, c - 45) for c in color)
    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color, outline=outline, width=2)
    elif shape == "rect":
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline, width=2)
    elif shape == "triangle":
        draw.polygon([(x0 + w / 2, y0), (x0, y1), (x1, y1)], fill=color, outline=outline)
    elif shape == "diamond":
        draw.polygon(
            [(x0 + w / 2, y0), (x1, y0 + h / 2), (x0 + w / 2, y1), (x0, y0 + h / 2)],
            fill=color,
            outline=outline,
        )
    elif shape == "trapezoid":
        draw.polygon(
            [(x0 + w * 0.2, y0), (x1 - w * 0.2, y0), (x1, y1), (x0, y1)],
            fill=color,
            outline=outline,
        )
    elif shape == "arc":
        draw.pieslice([x0, y0 - h, x1, y1], start=20, end=160, fill=color, outline=outline)
    elif shape == "bottle":
        draw.rectangle([x0 + w * 0.35, y0, x1 - w * 0.35, y0 + h * 0.3], fill=color)
        draw.rounded_rectangle(
            [x0, y0 + h * 0.28, x1, y1], radius=int(w * 0.2), fill=color, outline=outline, width=2
        )
    elif shape == "wheels":
        r = min(w, h) * 0.32
        draw.ellipse([x0, y1 - 2 * r, x0 + 2 * r, y1], outline=color, width=4)
        draw.ellipse([x1 - 2 * r, y1 - 2 * r, x1, y1], outline=color, width=4)
        draw.line([x0 + r, y1 - r, x0 + w / 2, y0 + h * 0.2, x1 - r, y1 - r], fill=color, width=4)
    elif shape == "person":
        head_r = w * 0.22
        cx = x0 + w / 2
        draw.ellipse([cx - head_r, y0, cx + head_r, y0 + 2 * head_r], fill=color)
        draw.rounded_rectangle(
            [x0 + w * 0.25, y0 + 2 * head_r, x1 - w * 0.25, y1],
            radius=int(w * 0.15),
            fill=color,
        )
    else:  # blob
        draw.ellipse([x0, y0 + h * 0.15, x1, y1], fill=color, outline=outline, width=2)
        draw.ellipse([x0 + w * 0.6, y0, x1, y0 + h * 0.45], fill=color)


def _place_boxes(rng, n: int) -> list[tuple[float, float, float, float]]:
    """Non-overlapping placements on a coarse grid.

    Non-overlap is a requirement, not an aesthetic: the low-resource crop in
    Axis 2 crops to a fraction of the *answer object's* area, and overlapping
    instances would make the resulting percept ambiguous in an uncontrolled way.
    """
    cols, rows = 3, 2
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    cw, ch = CANVAS[0] / cols, CANVAS[1] / rows
    boxes = []
    for c, r in cells[:n]:
        w = rng.uniform(0.45, 0.70) * cw
        h = rng.uniform(0.45, 0.70) * ch
        x = c * cw + rng.uniform(0.08, 0.92) * (cw - w)
        y = r * ch + rng.uniform(0.08, 0.92) * (ch - h)
        boxes.append((round(x, 1), round(y, 1), round(w, 1), round(h, 1)))
    return boxes


def _spatial_predicate(a: list[float], b: list[float]) -> str:
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    if abs(ax - bx) >= abs(ay - by):
        return "to the left of" if ax < bx else "to the right of"
    return "above" if ay < by else "below"


@register_source("synthetic")
def load_synthetic(
    limit: int = 5000, out_dir: Path | None = None, seed_base: int = 20260721, **_
) -> Iterator[SceneGraph]:
    """Render ``limit`` scenes and yield their exact scene graphs."""
    out_dir = Path(out_dir or IMAGES_CLEAN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(CATEGORIES)
    colors = list(COLORS)

    for i in range(limit):
        image_id = f"syn{i:07d}"
        rng = rng_for("synthetic", image_id, base=seed_base)
        n_obj = rng.randint(3, 5)

        # 'person' is drawn first when present so held-object relations have a
        # subject; otherwise relations fall back to spatial predicates.
        picked = rng.sample(names, n_obj)
        if rng.random() < 0.5 and "person" not in picked:
            picked[0] = "person"

        boxes = _place_boxes(rng, n_obj)
        img = Image.new("RGB", CANVAS, rng.choice(BACKGROUNDS))
        draw = ImageDraw.Draw(img)

        objects: list[Obj] = []
        for j, (name, box) in enumerate(zip(picked, boxes)):
            color = rng.choice(colors)
            _draw_shape(draw, CATEGORIES[name], box, COLORS[color])
            objects.append(
                Obj(
                    name=name,
                    bbox=list(box),
                    attributes=[color],
                    object_id=f"{image_id}-o{j}",
                )
            )

        relations: list[Relation] = []
        person = next((o for o in objects if o.name == "person"), None)
        others = [o for o in objects if o is not person]
        if person and others:
            held = rng.choice(others)
            relations.append(
                Relation(
                    subject="person",
                    predicate=rng.choice(HELD_PREDICATES),
                    obj=held.name,
                    subject_bbox=person.bbox,
                    object_bbox=held.bbox,
                )
            )
        for a, b in zip(objects, objects[1:]):
            relations.append(
                Relation(
                    subject=a.name,
                    predicate=_spatial_predicate(a.bbox, b.bbox),
                    obj=b.name,
                    subject_bbox=a.bbox,
                    object_bbox=b.bbox,
                )
            )

        path = out_dir / f"{image_id}.jpg"
        if not path.exists():
            img.save(path, quality=95)

        yield SceneGraph(
            image_id=image_id,
            image_path=str(path),
            width=CANVAS[0],
            height=CANVAS[1],
            source="synthetic",
            objects=objects,
            relations=relations,
        )
