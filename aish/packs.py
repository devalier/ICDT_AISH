"""Read-only access to the editable pack files for the web layer.

The packs are the Working Group's source of truth (see ``packs/README.md``). The
app renders them; it never writes them. Definitions are cached with the file
mtimes as the key, so an edit on disk shows up without a restart but re-parsing
does not happen on every request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.loader import Packs, load, packs_dir
from harness.validate import Problem, validate


@dataclass(frozen=True)
class PackView:
    packs: Packs
    problems: tuple[Problem, ...]
    signature: tuple

    @property
    def has_errors(self) -> bool:
        return any(p.severity == "error" for p in self.problems)

    def suites(self) -> list[dict]:
        rows = []
        for path, suite in self.packs.suites.items():
            rows.append({**suite, "_path": path})
        return sorted(rows, key=lambda s: (s.get("family", ""), s.get("id", "")))

    def suite(self, suite_id: str) -> dict | None:
        for suite in self.suites():
            if suite.get("id") == suite_id:
                return suite
        return None

    def criteria(self) -> dict:
        return self.packs.criteria.get("criteria") or {}

    def composite(self) -> dict:
        return self.packs.criteria.get("composite") or {}

    def attributes(self) -> list[dict]:
        return self.packs.attributes.get("attributes") or []

    def fleet(self) -> list[dict]:
        return self.packs.models.get("models") or []

    def judges(self) -> list[dict]:
        return self.packs.models.get("judges") or []


def _signature(root: Path) -> tuple:
    files = sorted(root.rglob("*.yaml"))
    return tuple((str(p), p.stat().st_mtime_ns) for p in files)


_cache: PackView | None = None


def current() -> PackView:
    global _cache
    root = packs_dir()
    signature = _signature(root)
    if _cache is not None and _cache.signature == signature:
        return _cache
    packs = load(root)
    _cache = PackView(packs=packs, problems=tuple(validate(packs)), signature=signature)
    return _cache


_euiba_cache: tuple[tuple, list[dict]] | None = None


def euiba_groups() -> list[dict]:
    """The registration drop-list, grouped for an ``<optgroup>`` select."""
    global _euiba_cache
    path = packs_dir() / "euibas.yaml"
    signature = ((str(path), path.stat().st_mtime_ns),)
    if _euiba_cache is not None and _euiba_cache[0] == signature:
        return _euiba_cache[1]
    with path.open(encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle) or {}
    groups = data.get("groups") or []
    _euiba_cache = (signature, groups)
    return groups


def euiba_lookup() -> dict[str, str]:
    return {
        entry["id"]: entry["name"]
        for group in euiba_groups()
        for entry in group.get("entries", [])
    }
