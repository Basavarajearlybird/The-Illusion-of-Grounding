"""Provenance: a flat, addressable table of every number the paper may quote.

`make figures` writes ``paper_numbers.json`` — a nested dict of every scalar the
analysis produced. This module flattens it into dotted keys (``fig1_decay.
sim-open-7b.decay_lambda``) so a manuscript can reference a number by key instead
of transcribing its value, and so the audit gate can check that every referenced
key actually exists and resolves to the value printed.

The point, in the words of the build directive: *the paper can never still say
"ADS = 0.87" if the rebuild measured something else.* A number that is pulled
from provenance at render time cannot go stale; a number typed by hand can, and
the audit is what catches it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gib.config import FIGURES_DIR


def flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten nested dicts/lists into dotted-key -> scalar."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}.{i}", out)
    else:
        out[prefix] = obj
    return out


def load_paper_numbers(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or FIGURES_DIR / "paper_numbers.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `make figures` first so provenance exists."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_provenance(path: Path | None = None) -> dict[str, Any]:
    """The full flat key -> value table."""
    return flatten(load_paper_numbers(path))


def resolve(key: str, table: dict[str, Any]) -> Any:
    """Look up a dotted key, raising a helpful error listing near-misses.

    A key that points at a list (``models`` when the table holds ``models.0``,
    ``models.1``, …) resolves to the comma-joined elements, so a manuscript can
    reference a whole list by its base key.
    """
    if key in table:
        return table[key]

    # List-join: gather contiguous integer-indexed children.
    elems = []
    i = 0
    while f"{key}.{i}" in table:
        elems.append(table[f"{key}.{i}"])
        i += 1
    if elems:
        return ", ".join(str(e) for e in elems)

    # Suggest candidates sharing the longest prefix, to make typos obvious.
    parts = key.split(".")
    for depth in range(len(parts), 0, -1):
        pfx = ".".join(parts[:depth])
        near = [k for k in table if k.startswith(pfx)]
        if near:
            raise KeyError(
                f"provenance key {key!r} not found. "
                f"{len(near)} key(s) share prefix {pfx!r}, e.g. {sorted(near)[:5]}"
            )
    raise KeyError(f"provenance key {key!r} not found and no prefix matches.")


def write_provenance_csv(out: Path | None = None, path: Path | None = None) -> Path:
    """Emit ``provenance.csv`` — one row per (key, value, type). Released with the
    paper so every quoted number is independently auditable."""
    table = build_provenance(path)
    out = Path(out or FIGURES_DIR / "provenance.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value", "type"])
        for k in sorted(table):
            v = table[k]
            w.writerow([k, v, type(v).__name__])
    return out
