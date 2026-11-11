"""Data model and exception types for st_analyze (Stage 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class StAnalyzeError(Exception):
    """Raised for unrecoverable Stage-2 errors (bad/missing manifest, etc.)."""


@dataclass
class AnalyzeResult:
    """Outcome of analyzing one .st source through the vendored analyzer."""

    ast_xml: str
    invariants: list[dict[str, Any]]
    pdg_dot: str
    pdg_structured: dict[str, Any]
    programs: list[str] = field(default_factory=list)
    state_variable: str | None = None
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    pdg_components: list[dict[str, Any]] = field(default_factory=list)
