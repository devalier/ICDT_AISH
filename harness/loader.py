"""Load the editable pack files.

The loader does no interpretation beyond reading YAML and locating files. Validation
lives in :mod:`harness.validate` so that loading a malformed pack still produces a
structure the validator can report specific errors about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def packs_dir() -> Path:
    override = os.environ.get("EUIBA_PACKS_DIR")
    return Path(override) if override else repo_root() / "packs"


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class Packs:
    """Everything the Working Group owns, as loaded from disk."""

    root: Path
    models: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)
    suites: dict = field(default_factory=dict)  # path -> suite mapping

    @property
    def model_ids(self) -> set[str]:
        return {m.get("id") for m in self.models.get("models", []) or []}

    @property
    def cloud_model_ids(self) -> set[str]:
        return {
            m.get("id")
            for m in self.models.get("models", []) or []
            if m.get("deployment") == "cloud"
        }

    @property
    def attribute_ids(self) -> set[str]:
        return {a.get("id") for a in self.attributes.get("attributes", []) or []}

    @property
    def criteria_names(self) -> set[str]:
        return set((self.criteria.get("criteria") or {}).keys())

    @property
    def judge_ids(self) -> set[str]:
        return {j.get("id") for j in self.models.get("judges", []) or []}


def load(root: Path | None = None) -> Packs:
    root = root or packs_dir()
    packs = Packs(root=root)

    for name, attr in (
        ("models.yaml", "models"),
        ("attributes.yaml", "attributes"),
        ("criteria.yaml", "criteria"),
    ):
        path = root / name
        if path.exists():
            setattr(packs, attr, read_yaml(path) or {})

    suites_dir = root / "suites"
    if suites_dir.is_dir():
        for path in sorted(suites_dir.glob("*.yaml")):
            packs.suites[str(path.relative_to(root))] = read_yaml(path) or {}

    return packs
