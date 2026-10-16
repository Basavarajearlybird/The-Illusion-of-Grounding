"""Layer 2.1 -- the model zoo."""

from gib.adapters.base import BaseAdapter, GenerationResult, ModelAdapter
from gib.adapters.registry import available_models, get_adapter, register_adapter

__all__ = [
    "BaseAdapter",
    "GenerationResult",
    "ModelAdapter",
    "available_models",
    "get_adapter",
    "register_adapter",
]
