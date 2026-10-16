"""Source ingestion (Layer 1.1).

Every source normalises into the same :class:`~gib.dataengine.scene.SceneGraph`
IR, so the QA generator, the mutators and the conflict generator are written once
against structured annotations rather than three times against three formats.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from gib.dataengine.scene import SceneGraph

_REGISTRY: dict[str, Callable[..., Iterator[SceneGraph]]] = {}


def register_source(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def get_source(name: str) -> Callable[..., Iterator[SceneGraph]]:
    if name not in _REGISTRY:
        # Import lazily so a missing optional source never breaks the package.
        from gib.dataengine.sources import coco, synthetic, visual_genome  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_sources() -> list[str]:
    from gib.dataengine.sources import coco, synthetic, visual_genome  # noqa: F401

    return sorted(_REGISTRY)
