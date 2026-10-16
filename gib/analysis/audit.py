"""The paper-audit gate.

Two capabilities, both aimed at one guarantee: **no number in the manuscript can
silently disagree with what the pipeline measured.**

* ``render`` — fill a manuscript *template* whose numeric claims are placeholders
  ``{{provenance.key:format}}`` with values pulled live from
  ``paper_numbers.json``. A rendered number literally cannot be stale, because it
  is produced from provenance at render time.

* ``audit`` — scan a manuscript for placeholders (and for an optional inline
  audit syntax ``VALUE⟨provenance.key⟩``) and verify every referenced key exists
  and, for the inline form, that the printed value matches provenance within a
  tolerance. Exit non-zero on any mismatch so it can gate a release.

Wire it into ``make paper-audit``. Run it before any public release.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from gib.analysis.provenance import build_provenance, resolve

# {{ key.path }} or {{ key.path : .3f }}
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*(?::\s*([^}]+?)\s*)?\}\}")
# printed_value⟨key.path⟩  — an already-rendered number tagged with its source
INLINE_AUDIT = re.compile(r"([-+]?\d[\d,]*\.?\d*)\s*⟨\s*([A-Za-z0-9_.\-]+)\s*⟩")
# HTML comments hold documentation that may show literal placeholder syntax; they
# are stripped before auditing so a usage example is not mistaken for a claim.
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_comments(text: str) -> str:
    return HTML_COMMENT.sub("", text)


def _format(value, spec: str | None) -> str:
    if spec:
        try:
            return format(value, spec)
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render(text: str, table: dict) -> tuple[str, list[str]]:
    """Replace every placeholder with its provenance value. Returns (text, errors)."""
    errors: list[str] = []

    def _sub(m: re.Match) -> str:
        key, spec = m.group(1), m.group(2)
        try:
            val = resolve(key, table)
        except KeyError as exc:
            errors.append(str(exc))
            return m.group(0)
        return _format(val, spec)

    return PLACEHOLDER.sub(_sub, text), errors


@dataclass
class AuditResult:
    ok: bool
    n_placeholders: int = 0
    n_inline_checked: int = 0
    unresolved: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"paper-audit: {'PASS' if self.ok else 'FAIL'}",
            f"  placeholders referenced: {self.n_placeholders}",
            f"  inline values checked:   {self.n_inline_checked}",
        ]
        for u in self.unresolved:
            lines.append(f"  UNRESOLVED: {u}")
        for m in self.mismatches:
            lines.append(f"  MISMATCH:   {m}")
        return "\n".join(lines)


def audit_text(text: str, table: dict, *, rel_tol: float = 1e-3) -> AuditResult:
    """Check every provenance reference in ``text``."""
    text = _strip_comments(text)
    unresolved: list[str] = []
    mismatches: list[str] = []

    placeholders = PLACEHOLDER.findall(text)
    for key, _spec in placeholders:
        try:
            resolve(key, table)
        except KeyError as exc:
            unresolved.append(str(exc))

    inline = INLINE_AUDIT.findall(text)
    for printed, key in inline:
        try:
            actual = resolve(key, table)
        except KeyError as exc:
            unresolved.append(str(exc))
            continue
        try:
            printed_val = float(printed.replace(",", ""))
            actual_val = float(actual)
        except (ValueError, TypeError):
            if str(printed).strip() != str(actual).strip():
                mismatches.append(f"{key}: manuscript {printed!r} != provenance {actual!r}")
            continue
        tol = max(rel_tol * abs(actual_val), rel_tol)
        if not math.isclose(printed_val, actual_val, abs_tol=tol, rel_tol=rel_tol):
            mismatches.append(
                f"{key}: manuscript prints {printed_val} but provenance says {actual_val}"
            )

    return AuditResult(
        ok=not unresolved and not mismatches,
        n_placeholders=len(placeholders),
        n_inline_checked=len(inline),
        unresolved=unresolved,
        mismatches=mismatches,
    )


def audit_paths(paths: list[Path], numbers_path: Path | None = None, *, rel_tol: float = 1e-3):
    """Audit one or more manuscript files against the provenance table."""
    table = build_provenance(numbers_path)
    combined = AuditResult(ok=True)
    for p in paths:
        p = Path(p)
        if not p.exists():
            combined.ok = False
            combined.unresolved.append(f"manuscript not found: {p}")
            continue
        r = audit_text(p.read_text(encoding="utf-8"), table, rel_tol=rel_tol)
        combined.n_placeholders += r.n_placeholders
        combined.n_inline_checked += r.n_inline_checked
        combined.unresolved += [f"{p.name}: {u}" for u in r.unresolved]
        combined.mismatches += [f"{p.name}: {m}" for m in r.mismatches]
        combined.ok = combined.ok and r.ok
    return combined, table


def render_paths(template: Path, out: Path, numbers_path: Path | None = None) -> list[str]:
    """Render a manuscript template to ``out``, returning any unresolved keys."""
    table = build_provenance(numbers_path)
    text = Path(template).read_text(encoding="utf-8")
    rendered, errors = render(text, table)
    Path(out).write_text(rendered, encoding="utf-8")
    return errors
