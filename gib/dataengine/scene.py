"""The scene-graph intermediate representation.

COCO gives instance inventories, Visual Genome gives attributes and
``(subject, predicate, object)`` triples, and the synthetic source gives both by
construction. Normalising all three into one IR is what lets 1.2's mutators
operate *structurally* -- swapping a predicate or an attribute is a field
assignment, not a regex over prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Obj:
    """One annotated object instance."""

    name: str
    bbox: list[float]  # [x, y, w, h] in pixels
    attributes: list[str] = field(default_factory=list)
    object_id: str = ""

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2]) * max(0.0, self.bbox[3])


@dataclass
class Relation:
    """One ``(subject, predicate, object)`` triple."""

    subject: str
    predicate: str
    obj: str
    subject_bbox: list[float]
    object_bbox: list[float]


@dataclass
class SceneGraph:
    image_id: str
    image_path: str
    width: int
    height: int
    source: str
    objects: list[Obj] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    def objects_named(self, name: str) -> list[Obj]:
        return [o for o in self.objects if o.name == name]

    def unique_objects(self) -> list[Obj]:
        """Objects whose category appears exactly once.

        Uniqueness matters: "what colour is the umbrella?" is only
        machine-verifiable when there is one umbrella. Ambiguous scenes are
        filtered here rather than being caught later by the round-trip check.
        """
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.name] = counts.get(o.name, 0) + 1
        return [o for o in self.objects if counts[o.name] == 1]

    def largest_object(self) -> Obj | None:
        return max(self.objects, key=lambda o: o.area, default=None)
