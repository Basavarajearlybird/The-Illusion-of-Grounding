"""Config loading and repo paths.

Runs are config-driven (plain YAML, Hydra-compatible layout) so that a run is
fully described by {model, dataset config, axis, dose list, sampling params} and
can be replayed from the logged config alone.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = Path(os.environ.get("GIB_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("GIB_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("GIB_FIGURES_DIR", REPO_ROOT / "figures"))
CACHE_DIR = Path(os.environ.get("GIB_CACHE_DIR", REPO_ROOT / ".cache"))

# Layer 1 outputs
RAW_DIR = DATA_DIR / "raw"
IMAGES_CLEAN_DIR = DATA_DIR / "images" / "clean"
IMAGES_DEGRADED_DIR = DATA_DIR / "images" / "degraded"
PAIRS_PATH = DATA_DIR / "gib15k" / "clean_pairs.parquet"
VARIANTS_DIR = DATA_DIR / "gib15k" / "variants"
RELEASE_DIR = DATA_DIR / "release"

# Layer 3 outputs
INTERP_DIR = RESULTS_DIR / "interp"
# Layer 4 outputs
HUMAN_DIR = RESULTS_DIR / "human"


def ensure_dirs() -> None:
    for d in (
        RAW_DIR,
        IMAGES_CLEAN_DIR,
        IMAGES_DEGRADED_DIR,
        VARIANTS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        CACHE_DIR,
        INTERP_DIR,
        HUMAN_DIR,
        RELEASE_DIR,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def git_sha() -> str:
    """Logged with every run. Returns 'nogit' outside a repository so that the
    provenance columns never crash a run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------- #
# Typed views over the dataset config
# --------------------------------------------------------------------------- #


@dataclass
class NoiseConfig:
    doses: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    embedder: str = "BAAI/bge-large-en-v1.5"
    hard_negative_ratio: float = 0.5  # share drawn from surgical mutation
    top_k_neighbours: int = 32


@dataclass
class VisualConfig:
    gaussian_sigma: list[float] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    motion_kernel: list[int] = field(default_factory=lambda: [5, 10, 20, 40])
    crop_visible: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.10, 0.05])
    native_resolution: int = 336


@dataclass
class ConflictConfig:
    make_twins: bool = True


@dataclass
class CombinedConfig:
    """The only place the architecture crosses two axes -- it powers the
    vulnerability heatmap and is deliberately kept to a small subset."""

    n_pairs: int = 1000
    sigma_grid: list[float] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    distractor_grid: list[int] = field(default_factory=lambda: [0, 1, 3, 5, 10])


@dataclass
class DatasetConfig:
    n_pairs: int = 15000
    per_sub_type: int = 5000
    sources: list[str] = field(default_factory=lambda: ["coco", "visual_genome"])
    seed: int = 20260721
    eval_split_size: int = 50000
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    combined: CombinedConfig = field(default_factory=CombinedConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "dataset.yaml") -> DatasetConfig:
        raw = load_yaml(path)
        return cls(
            n_pairs=raw.get("n_pairs", 15000),
            per_sub_type=raw.get("per_sub_type", 5000),
            sources=raw.get("sources", ["coco", "visual_genome"]),
            seed=raw.get("seed", 20260721),
            eval_split_size=raw.get("eval_split_size", 50000),
            noise=NoiseConfig(**raw.get("noise", {})),
            visual=VisualConfig(**raw.get("visual", {})),
            conflict=ConflictConfig(**raw.get("conflict", {})),
            combined=CombinedConfig(**raw.get("combined", {})),
        )
