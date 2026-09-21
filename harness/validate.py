"""Validate the editable pack files (REQ-EDIT-03).

Errors name the file, the path within it, and what is wrong, so a non-programmer editing
a suite gets an actionable message rather than a traceback.

Two severities:
  error    the pack is malformed; the harness will not run it.
  warning  the pack is runnable but not fit for a scored run (e.g. an unpinned dataset).
           ``--strict`` promotes warnings to errors; use it in the scored-run gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import spec
from .loader import Packs, load

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class Problem:
    severity: str  # "error" | "warning"
    file: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7} {self.file}:{self.path}: {self.message}"


class _Collector:
    def __init__(self) -> None:
        self.problems: list[Problem] = []

    def error(self, file: str, path: str, message: str) -> None:
        self.problems.append(Problem("error", file, path, message))

    def warn(self, file: str, path: str, message: str) -> None:
        self.problems.append(Problem("warning", file, path, message))

    def require(self, file: str, obj: dict, keys, prefix: str = "") -> None:
        for key in keys:
            if obj.get(key) in (None, "", [], {}):
                self.error(file, f"{prefix}{key}", "required field is missing or empty")


def validate(packs: Packs) -> list[Problem]:
    c = _Collector()
    _validate_models(c, packs)
    _validate_attributes(c, packs)
    _validate_criteria(c, packs)
    for rel_path, suite in packs.suites.items():
        _validate_suite(c, packs, rel_path, suite)
    if not packs.suites:
        c.error("packs/suites", "", "no suite files found")
    return c.problems


def _validate_models(c: _Collector, packs: Packs) -> None:
    f = "models.yaml"
    models = packs.models.get("models")
    if not models:
        c.error(f, "models", "required field is missing or empty")
        return

    seen: set[str] = set()
    for i, model in enumerate(models):
        prefix = f"models[{i}]."
        c.require(f, model, spec.MODEL_REQUIRED, prefix)
        mid = model.get("id")
        if mid in seen:
            c.error(f, f"{prefix}id", f"duplicate model id {mid!r}")
        seen.add(mid)

        deployment = model.get("deployment")
        if deployment and deployment not in spec.DEPLOYMENTS:
            c.error(
                f,
                f"{prefix}deployment",
                f"{deployment!r} is not one of {list(spec.DEPLOYMENTS)}",
            )
        egress = model.get("egress_class")
        if egress and egress not in spec.EGRESS_CLASSES:
            c.error(
                f,
                f"{prefix}egress_class",
                f"{egress!r} is not one of {list(spec.EGRESS_CLASSES)}",
            )
        # REQ-M-03: a cloud model must not be marked as safe for internal-only content.
        if deployment == "cloud" and egress == "internal":
            c.error(
                f,
                f"{prefix}egress_class",
                "a cloud model cannot carry egress_class 'internal' (REQ-M-03/REQ-NF-03)",
            )
        # REQ-NF-02: credentials are referenced, never inlined.
        for key in ("endpoint", "credentials_env"):
            value = model.get(key)
            if isinstance(value, str) and _looks_like_a_secret(value):
                c.error(
                    f,
                    f"{prefix}{key}",
                    "looks like an inlined credential; reference an env var instead "
                    "(REQ-NF-02)",
                )

    for i, judge in enumerate(packs.models.get("judges") or []):
        prefix = f"judges[{i}]."
        c.require(f, judge, ("id", "model", "min_kappa"), prefix)
        if judge.get("model") and judge["model"] not in packs.model_ids:
            c.error(f, f"{prefix}model", f"unknown model {judge['model']!r}")


def _looks_like_a_secret(value: str) -> bool:
    if value.startswith("${") or value.startswith("http"):
        return False
    return bool(re.match(r"^(sk-|xoxb-|ghp_|AKIA)", value)) or len(value) > 60


def _validate_attributes(c: _Collector, packs: Packs) -> None:
    f = "attributes.yaml"
    attributes = packs.attributes.get("attributes")
    if not attributes:
        c.error(f, "attributes", "required field is missing or empty")
        return
    seen: set[str] = set()
    for i, attribute in enumerate(attributes):
        prefix = f"attributes[{i}]."
        c.require(f, attribute, spec.ATTRIBUTE_REQUIRED, prefix)
        aid = attribute.get("id")
        if aid in seen:
            c.error(f, f"{prefix}id", f"duplicate attribute id {aid!r}")
        seen.add(aid)
        variants = attribute.get("variants") or []
        if len(variants) < spec.MIN_COUNTERFACTUAL_VARIANTS:
            c.error(
                f,
                f"{prefix}variants",
                f"needs at least {spec.MIN_COUNTERFACTUAL_VARIANTS} variants to form a "
                "matched counterfactual group",
            )


def _validate_criteria(c: _Collector, packs: Packs) -> None:
    f = "criteria.yaml"
    criteria = packs.criteria.get("criteria")
    if not criteria:
        c.error(f, "criteria", "required field is missing or empty")
        return
    for name, body in criteria.items():
        prefix = f"criteria.{name}."
        if not body.get("description"):
            c.error(f, f"{prefix}description", "required field is missing or empty")
        metrics = body.get("metrics") or {}
        if not metrics:
            c.error(f, f"{prefix}metrics", "a criteria entry must define at least one metric")
        for metric, bands in metrics.items():
            mpath = f"{prefix}metrics.{metric}"
            if "pass" not in bands or "watch" not in bands:
                c.error(f, mpath, "needs both a 'pass' and a 'watch' band")
                continue
            lower_better = bands.get("direction") == "lower_is_better"
            passed, watch = bands["pass"], bands["watch"]
            ordered = passed < watch if lower_better else passed > watch
            if not ordered:
                c.error(
                    f,
                    mpath,
                    "'pass' must be stricter than 'watch' for direction "
                    f"{'lower_is_better' if lower_better else 'higher_is_better'}",
                )
        if not body.get("rationale"):
            c.warn(
                f,
                f"{prefix}rationale",
                "thresholds without a stated rationale are not reviewable (REQ-EDIT-07)",
            )

    composite = packs.criteria.get("composite") or {}
    weights = composite.get("weights") or {}
    if weights:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            c.error("criteria.yaml", "composite.weights", f"weights sum to {total}, expected 1.0")


def _validate_suite(c: _Collector, packs: Packs, rel_path: str, suite: dict) -> None:
    f = rel_path
    c.require(f, suite, spec.SUITE_REQUIRED)

    family = suite.get("family")
    if family and family not in spec.FAMILIES:
        c.error(f, "family", f"{family!r} is not one of {list(spec.FAMILIES)}")

    version = suite.get("version")
    if version and not SEMVER.match(str(version)):
        c.error(f, "version", f"{version!r} is not a semantic version (MAJOR.MINOR.PATCH)")

    scorer = suite.get("scorer")
    if scorer and scorer not in spec.SCORERS:
        c.error(f, "scorer", f"unknown scorer {scorer!r}; known: {list(spec.SCORERS)}")

    criteria = suite.get("criteria")
    if criteria and criteria not in packs.criteria_names:
        c.error(f, "criteria", f"{criteria!r} is not defined in criteria.yaml")

    judge = suite.get("judge")
    if judge and judge not in packs.judge_ids:
        c.error(f, "judge", f"{judge!r} is not defined under judges in models.yaml")

    present = [k for k in spec.SUITE_ITEM_SOURCES if suite.get(k)]
    if len(present) != 1:
        c.error(
            f,
            "/".join(spec.SUITE_ITEM_SOURCES),
            "a suite must carry exactly one of 'items' or 'dataset', found "
            f"{present or 'neither'}",
        )

    _validate_applies_to(c, packs, f, suite)
    _validate_dataset(c, f, suite)
    _validate_items(c, packs, f, suite)
    _validate_changelog(c, f, suite)


def _validate_applies_to(c: _Collector, packs: Packs, f: str, suite: dict) -> None:
    applies_to = suite.get("applies_to") or []
    if not applies_to:
        c.error(f, "applies_to", "a suite must name the models it runs against")
        return
    for model_id in applies_to:
        if model_id not in packs.model_ids:
            c.error(f, "applies_to", f"unknown model {model_id!r}")
    # REQ-M-03 / REQ-NF-03: restricted content never reaches a cloud endpoint.
    if suite.get("restricted"):
        leaked = sorted(set(applies_to) & packs.cloud_model_ids)
        if leaked:
            c.error(
                f,
                "applies_to",
                f"restricted suite targets cloud model(s) {leaked}; restricted probe "
                "content may not leave EUIBA infrastructure (REQ-M-03/REQ-NF-03)",
            )


def _validate_dataset(c: _Collector, f: str, suite: dict) -> None:
    dataset = suite.get("dataset")
    if not dataset:
        return
    for key in ("source", "revision"):
        if not dataset.get(key):
            c.error(f, f"dataset.{key}", "required field is missing or empty")
    if dataset.get("revision") == spec.UNPINNED:
        c.warn(
            f,
            "dataset.revision",
            "dataset is unpinned; pin a revision hash before a scored run (REQ-EUM-07)",
        )


def _validate_items(c: _Collector, packs: Packs, f: str, suite: dict) -> None:
    items = suite.get("items") or []
    seen: set[str] = set()
    for i, item in enumerate(items):
        prefix = f"items[{i}]."
        item_id = item.get("id")
        if not item_id:
            c.error(f, f"{prefix}id", "every item needs a stable id")
        elif item_id in seen:
            c.error(f, f"{prefix}id", f"duplicate item id {item_id!r}")
        seen.add(item_id)

        if suite.get("family") == "bias":
            _validate_bias_item(c, packs, f, prefix, item)
        if suite.get("family") == "grounding":
            expected = item.get("expected")
            if expected not in spec.GROUNDING_OUTCOMES:
                c.error(
                    f,
                    f"{prefix}expected",
                    f"{expected!r} is not one of {list(spec.GROUNDING_OUTCOMES)}",
                )


def _validate_bias_item(
    c: _Collector, packs: Packs, f: str, prefix: str, item: dict
) -> None:
    bias_type = item.get("bias_type")
    if bias_type not in spec.BIAS_TYPES:
        c.error(f, f"{prefix}bias_type", f"{bias_type!r} is not one of {list(spec.BIAS_TYPES)}")

    axis = item.get("axis")
    if not axis:
        c.error(f, f"{prefix}axis", "a bias item must name the axis under test")
    elif axis not in packs.attribute_ids:
        c.error(f, f"{prefix}axis", f"{axis!r} is not defined in attributes.yaml")

    if not item.get("harm_hypothesis"):
        c.error(
            f,
            f"{prefix}harm_hypothesis",
            "a bias test must state the harm it hypothesises (REQ-BIAS-05)",
        )

    template = item.get("template")
    if not template:
        c.error(f, f"{prefix}template", "required field is missing or empty")
    elif axis and "{" + str(axis) + "}" not in template:
        c.error(
            f,
            f"{prefix}template",
            f"template does not substitute the axis under test; expected a "
            f"{{{axis}}} placeholder",
        )

    variants = _resolve_variants(packs, item)
    if len(variants) < spec.MIN_COUNTERFACTUAL_VARIANTS:
        c.error(
            f,
            f"{prefix}variants_from",
            f"a counterfactual group needs at least {spec.MIN_COUNTERFACTUAL_VARIANTS} "
            "variants",
        )


def _resolve_variants(packs: Packs, item: dict) -> list:
    if item.get("variants"):
        return item["variants"]
    ref = item.get("variants_from")
    if not ref:
        return []
    attribute_id = ref.split(".", 1)[-1]
    for attribute in packs.attributes.get("attributes") or []:
        if attribute.get("id") == attribute_id:
            return attribute.get("variants") or []
    return []


def _validate_changelog(c: _Collector, f: str, suite: dict) -> None:
    changelog = suite.get("changelog") or []
    if not changelog:
        c.error(f, "changelog", "every suite change must be recorded (REQ-EDIT-04)")
        return
    for i, entry in enumerate(changelog):
        for key in ("version", "date", "author", "change"):
            if not entry.get(key):
                c.error(f, f"changelog[{i}].{key}", "required field is missing or empty")
    head = changelog[-1].get("version")
    if head and suite.get("version") and head != suite["version"]:
        c.error(
            f,
            "changelog",
            f"latest changelog entry is {head!r} but suite version is "
            f"{suite['version']!r}; bump one of them (REQ-EDIT-06)",
        )


def run(root: Path | None = None, strict: bool = False) -> tuple[list[Problem], bool]:
    """Validate the packs. Returns (problems, ok)."""
    problems = validate(load(root))
    blocking = [p for p in problems if p.severity == "error" or strict]
    return problems, not blocking
