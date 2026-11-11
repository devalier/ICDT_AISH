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

    if len(suite.get("items") or []) > LIMITS["items"]:
        c.error(f, "items", f"at most {LIMITS['items']} items in one suite")
    if family == "capability":
        template = (suite.get("prompt") or {}).get("template", "")
        missing = [p for p in ("{question}", "{choice_0}", "{choice_1}", "{choice_2}", "{choice_3}")
                   if p not in template]
        if missing:
            c.error(f, "prompt.template", f"the prompt template must contain {', '.join(missing)}")
        if len(template) > LIMITS["template"]:
            c.error(f, "prompt.template", f"at most {LIMITS['template']} characters")
    if suite.get("probes_inline") is not None:
        _validate_probes(c, f, suite)
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
            "dataset is unpinned. Provision the data (scripts/fetch_eu_mmlu.py), then paste the "
            "revision it prints into dataset.revision before a scored run (REQ-EUM-07). "
            "Until then runs still record the data file's own revision.",
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
            if expected == "grounded" and not (
                item.get("reference_answer") or item.get("reference_answers")
            ):
                c.error(
                    f,
                    f"{prefix}reference_answers",
                    "an answerable item needs reference_answers; without one it cannot be "
                    "checked and would score as unparseable",
                )
            if item.get("source") and item.get("source_text"):
                c.error(f, f"{prefix}source_text", "use source (a file) or source_text (inline), not both")
            if item.get("source"):
                _validate_source_path(c, f, prefix, item["source"])
            if len(str(item.get("source_text") or "")) > LIMITS["source_text"]:
                c.error(f, f"{prefix}source_text", f"at most {LIMITS['source_text']} characters")
            if item.get("check_citation") and not (item.get("source") or item.get("source_text")):
                c.error(f, f"{prefix}source", "check_citation needs a supplied source to check against")
        if len(str(item.get("prompt") or item.get("template") or "")) > LIMITS["template"]:
            c.error(f, f"{prefix}prompt", f"at most {LIMITS['template']} characters")


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

    pools = item.get("pools")
    if pools is not None:
        _validate_pools(c, f, prefix, item, pools)

    template = item.get("template")
    by_variant = item.get("template_by_variant")
    if pools:
        # Name-based: the group is implied by a name drawn from the pool, so the template
        # carries {name} instead of the axis placeholder.
        if not template or "{name}" not in template:
            c.error(f, f"{prefix}template", "a probe with name pools needs a template containing {name}")
        elif axis and "{" + str(axis) + "}" in template:
            c.warn(f, f"{prefix}template", "the template names the group outright as well as by "
                   "name; a name-based probe should leave the group implicit")
    elif by_variant:
        # Hand-translated prompts: the variable is the whole prompt, not a placeholder.
        if template:
            c.error(f, f"{prefix}template", "use template or template_by_variant, not both")
        if not all(str(v).strip() for v in by_variant.values()):
            c.error(f, f"{prefix}template_by_variant", "every variant needs a non-empty prompt")
    elif not template:
        c.error(f, f"{prefix}template", "required field is missing or empty")
    elif axis and "{" + str(axis) + "}" not in template:
        c.error(
            f,
            f"{prefix}template",
            f"template does not substitute the axis under test; expected a "
            f"{{{axis}}} placeholder",
        )

    _validate_scoring(c, f, prefix, item)

    variants = _resolve_variants(packs, item)
    if len(variants) < spec.MIN_COUNTERFACTUAL_VARIANTS:
        c.error(
            f,
            f"{prefix}variants_from",
            f"a counterfactual group needs at least {spec.MIN_COUNTERFACTUAL_VARIANTS} "
            "variants",
        )


LIMITS = {"template": 6000, "rubric": 3000, "items": 600, "pool_names": 60, "pool_groups": 40,
          "source_text": 30000, "prompt": 6000, "name": 80}


def _validate_probes(c: _Collector, f: str, suite: dict) -> None:
    probes = suite.get("probes_inline")
    if not isinstance(probes, dict):
        c.error(f, "probes_inline", "must contain controls and injection lists")
        return
    hazard_ids = {h.get("id") for h in suite.get("hazards") or []}
    injection_ids = {h.get("id") for h in suite.get("injection_hazards") or []}
    seen: set = set()
    for kind in ("controls", "injection"):
        entries = probes.get(kind) or []
        if len(entries) > LIMITS["items"]:
            c.error(f, f"probes_inline.{kind}", f"at most {LIMITS['items']} entries")
        for i, entry in enumerate(entries):
            path = f"probes_inline.{kind}[{i}]"
            eid = entry.get("id")
            if not eid:
                c.error(f, f"{path}.id", "every probe needs a stable id")
            elif eid in seen:
                c.error(f, f"{path}.id", f"duplicate probe id {eid!r}")
            seen.add(eid)
            if not str(entry.get("prompt") or "").strip():
                c.error(f, f"{path}.prompt", "required field is missing or empty")
            elif len(entry["prompt"]) > LIMITS["prompt"]:
                c.error(f, f"{path}.prompt", f"at most {LIMITS['prompt']} characters")
            allowed = hazard_ids | injection_ids if kind == "controls" else injection_ids
            if entry.get("hazard") not in allowed:
                c.error(f, f"{path}.hazard", f"{entry.get('hazard')!r} is not a known hazard for this list")
            if kind == "injection" and not str(entry.get("canary") or "").strip():
                c.error(f, f"{path}.canary", "an injection probe needs the canary string a leak would contain")


def _validate_source_path(c: _Collector, f: str, prefix: str, source) -> None:
    """A source file must be one of the packs' own, never a path that escapes the folder."""
    text = str(source)
    if text.startswith("/") or ".." in Path(text).parts or "\\" in text or not text.startswith("sources/"):
        c.error(f, f"{prefix}source", "must be a file under sources/ (no absolute paths or '..')")


def _validate_pools(c: _Collector, f: str, prefix: str, item: dict, pools) -> None:
    if not isinstance(pools, dict) or len(pools) < spec.MIN_COUNTERFACTUAL_VARIANTS:
        c.error(f, f"{prefix}pools", "pools must map at least two groups to lists of names")
        return
    if len(pools) > LIMITS["pool_groups"]:
        c.error(f, f"{prefix}pools", f"at most {LIMITS['pool_groups']} groups")
    for group, names in pools.items():
        gpath = f"{prefix}pools.{group}"
        if not isinstance(names, list) or not names or not all(
                isinstance(n, str) and n.strip() and len(n) <= LIMITS["name"] for n in names):
            c.error(f, gpath, f"needs a list of non-empty names (each at most {LIMITS['name']} characters)")
        elif len(names) > LIMITS["pool_names"]:
            c.error(f, gpath, f"at most {LIMITS['pool_names']} names per group")
        elif len(names) < 3:
            c.warn(f, gpath, "fewer than 3 names: one unusual name can drive the result; "
                   "use 5 or more per group")
    sizes = {len(n) for n in pools.values() if isinstance(n, list)}
    if len(sizes) > 1:
        c.warn(f, f"{prefix}pools", "groups have different numbers of names; use equal-sized pools "
               "so no group is sampled more thinly")


def _validate_scoring(c: _Collector, f: str, prefix: str, item: dict) -> None:
    scoring = item.get("scoring")
    kind = item.get("kind")
    if kind is not None and kind not in spec.BIAS_KINDS:
        c.error(f, f"{prefix}kind", f"{kind!r} is not one of {list(spec.BIAS_KINDS)}")
    if scoring is None:
        if kind == "loaded_premise":
            c.error(f, f"{prefix}scoring", "a loaded_premise probe needs scoring.method: judge "
                    "(only a judge can tell whether a response challenges the premise)")
        return
    method = scoring.get("method") if isinstance(scoring, dict) else None
    if method != "judge":
        c.error(f, f"{prefix}scoring.method", "only 'judge' is supported here; leave scoring out to "
                "use outcome_extraction")
        return
    rubric = str(scoring.get("rubric") or "").strip()
    if not rubric:
        c.error(f, f"{prefix}scoring.rubric", "required field is missing or empty")
    elif len(rubric) > LIMITS["rubric"]:
        c.error(f, f"{prefix}scoring.rubric", f"at most {LIMITS['rubric']} characters")
    labels = scoring.get("labels") or {}
    for need in ("favourable", "unfavourable"):
        if not str(labels.get(need) or "").strip():
            c.error(f, f"{prefix}scoring.labels.{need}", "required: the label the judge returns for this outcome")
    given = [str(v).strip().upper() for v in labels.values() if v]
    if len(given) != len(set(given)):
        c.error(f, f"{prefix}scoring.labels", "each outcome needs a different label")
    if item.get("outcome_extraction"):
        c.error(f, f"{prefix}outcome_extraction", "use either scoring (judge) or outcome_extraction, not both")


def _resolve_variants(packs: Packs, item: dict) -> list:
    if item.get("pools"):
        variants = list(item["pools"])
    elif item.get("template_by_variant"):
        variants = list(item["template_by_variant"])
    elif item.get("variants"):
        variants = list(item["variants"])
    else:
        ref = item.get("variants_from")
        variants = []
        if ref:
            attribute_id = ref.split(".", 1)[-1]
            for attribute in packs.attributes.get("attributes") or []:
                if attribute.get("id") == attribute_id:
                    variants = list(attribute.get("variants") or [])
    excluded = {str(v) for v in item.get("exclude_variants") or []}
    return [v for v in variants if str(v) not in excluded]


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
