"""Determinism helpers.

The adversarial transformation layer is specified as a pure function
``(clean_pair, config) -> adversarial_variant``. That only holds if randomness is
derived from stable identifiers rather than from a global RNG whose state depends
on iteration order. Everything here derives a seed from *content*, so a variant
generated in a partial re-run is byte-identical to one from a full run.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

GLOBAL_SEED = 20260721


def derive_seed(*parts: Any, base: int = GLOBAL_SEED) -> int:
    """Deterministic 63-bit seed from a base seed and arbitrary key parts."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(f"{base}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def rng_for(*parts: Any, base: int = GLOBAL_SEED) -> random.Random:
    return random.Random(derive_seed(*parts, base=base))


def np_rng_for(*parts: Any, base: int = GLOBAL_SEED) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts, base=base) % (2**32))


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """Global seeding for the layers that cannot be made purely functional
    (model inference, sampling)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
