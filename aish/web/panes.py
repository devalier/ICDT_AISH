"""The GUI panes: model comparison, the test-suite pages, and the methodology page.

Navigation is a workflow (Models Selected for Benchmarking -> Run tests -> Compare models) followed by the
test harness: methodology, then one page per test suite. Each suite page carries its
basis, the actual tests, the readiness checklist and the results, so there is nothing
to cross-reference between panes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from harness.loader import packs_dir

from .. import judge, scoring
from ..config import get_settings
from ..db import get_session
from ..models import ModelTarget, Run
from ..packs import current
from .deps import AuthContext, optional_user, require_user, templates

router = APIRouter()

FAMILIES = ("capability", "safety", "bias", "grounding")
FAMILY_LABEL = {"capability": "Language", "safety": "Safety", "bias": "Bias",
                "grounding": "Grounding"}

# The headline metric shown per family on the comparison pane. Full metric sets are
# on the run page; this is the one number each family leads with.
HEADLINE = {
    "capability": ("parity_floor", "Parity floor", "higher"),
    "safety": ("violation_rate", "Violation rate", "lower"),
    "bias": ("rate_disparity", "Max disparity", "lower"),
    "grounding": ("hallucination_rate", "Hallucination rate", "lower"),
}

VERDICT_LABEL = {"pass": "Pass", "watch": "Watch", "fail": "Fail",  # nosec B105
                 "inconclusive": "Inconclusive", "no_data": "No data"}
# Status is never colour alone: every verdict also carries a glyph and its word.
GLYPH = {"pass": "✓", "watch": "!", "fail": "✕", "inconclusive": "?", "no_data": "–"}  # nosec B105

GROUNDING_SEGMENTS = (
    ("grounded", "Grounded"),
    ("abstained_correctly", "Abstained correctly"),
    ("abstained_wrongly", "Abstained wrongly"),
    ("hallucinated", "Hallucinated"),
    ("unparseable", "Unparseable"),
)


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #


def nav_suites() -> list[dict]:
    """The suites under 'Test suites' in the sidebar, in axis order."""
    order = {family: i for i, family in enumerate(FAMILIES)}
    suites = sorted(current().suites(), key=lambda s: (order.get(s.get("family"), 99), s["id"]))
    return [
        {
            "id": s["id"],
            "label": s.get("menu_label") or FAMILY_LABEL.get(s.get("family"), s["id"]),
            "title": s.get("title") or s["id"],
        }
        for s in suites
    ]


templates.env.globals["nav_suites"] = nav_suites


@router.get("/")
def root(context: AuthContext | None = Depends(optional_user)):
    return RedirectResponse("/dashboard" if context else "/login", status_code=303)


# --------------------------------------------------------------------------- #
# Results data shared by the comparison page and the suite pages
# --------------------------------------------------------------------------- #


def _latest_runs(db: Session, user_id: int) -> dict[tuple[int, str], Run]:
    """Most recent completed run per (target, suite)."""
    runs = db.scalars(
        select(Run)
        .where(Run.user_id == user_id, Run.status == "done")
        .order_by(Run.finished_at.desc())
    )
    latest: dict[tuple[int, str], Run] = {}
    for run in runs:
        latest.setdefault((run.target_id, run.suite_id), run)
    return latest


def _bands(view, suite: dict | None, metric: str) -> dict:
    if not suite:
        return {}
    return (view.criteria().get(suite.get("criteria"), {}).get("metrics") or {}).get(metric) or {}


def _grade(value, bands: dict, interval=None) -> str:
    if value is None or not bands:
        return "no_data"
    return scoring.verdict(value, bands, tuple(interval) if interval else None)


def _fmt(value, direction: str) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if direction == "lower" else f"{value:.2f}"


def _entries(db: Session, user_id: int, family: str, suites: list[dict]) -> list[dict]:
    """One entry per model target for a family, with its latest run if it has one."""
    view = current()
    targets = list(db.scalars(
        select(ModelTarget).where(ModelTarget.user_id == user_id).order_by(ModelTarget.label)
    ))
    latest = _latest_runs(db, user_id)
    metric, _, direction = HEADLINE[family]
    out = []
    for target in targets:
        entry = {"target": target, "label": target.label, "model": target.model_name,
                 "run": None, "summary": {}, "suite": None, "value": None,
                 "verdict": "no_data", "overall": "no_data", "n": 0, "sample": False}
        for suite in suites:
            run = latest.get((target.id, suite["id"]))
            if run is None:
                continue
            summary = json.loads(run.summary_json or "{}")
            value = (summary.get("metrics") or {}).get(metric)
            entry.update(
                run=run, summary=summary, suite=suite, value=value,
                verdict=_grade(value, _bands(view, suite, metric),
                               (summary.get("intervals") or {}).get(metric)),
                overall=summary.get("verdict", "no_data"), n=summary.get("n", 0),
                sample=bool(summary.get("sample_mode", True)),
            )
            break
        out.append(entry)
    return out


# ---- axis bar chart ------------------------------------------------------- #

_W, _LABEL_X, _PLOT_X, _PLOT_W, _ROW_H, _TOP = 780, 0, 200, 400, 40, 26


def _nice_max(values: list[float], watch, direction: str) -> float:
    peak = max(values) if values else 0.0
    if direction == "higher":
        return max(1.0, math.ceil(peak * 10) / 10)
    floor = (watch * 2.0) if watch else 0.2
    raw = max(floor, peak * 1.15)
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if raw <= step * 4 or step == 1.0:
            return min(1.0, math.ceil(raw / step) * step)
    return 1.0


def axis_chart(family: str, entries: list[dict], bands: dict) -> dict:
    """Geometry for one axis' comparison chart. Computed here so the template stays
    markup and the CSP needs no inline styles."""
    metric, label, direction = HEADLINE[family]
    pass_at, watch_at = bands.get("pass"), bands.get("watch")
    have = [e for e in entries if e["run"] is not None and e["value"] is not None]
    maximum = _nice_max([e["value"] for e in have], watch_at, direction)

    def x(value: float) -> float:
        return round(_PLOT_X + max(0.0, min(1.0, value / maximum)) * _PLOT_W, 1)

    ranked = sorted(have, key=lambda e: e["value"], reverse=(direction == "higher"))
    ranked += [e for e in entries if e not in have]
    height = _TOP + len(ranked) * _ROW_H + 26

    zones = []
    if pass_at is not None and watch_at is not None:
        if direction == "lower":
            spans = [(0, pass_at, "pass"), (pass_at, watch_at, "watch"), (watch_at, maximum, "fail")]
        else:
            spans = [(0, watch_at, "fail"), (watch_at, pass_at, "watch"), (pass_at, maximum, "pass")]
        for lo, hi, cls in spans:
            if hi > lo:
                zones.append({"x": x(lo), "w": round(x(hi) - x(lo), 1), "cls": cls})

    ticks = []
    for value in sorted({0.0, maximum, *(v for v in (pass_at, watch_at) if v is not None)}):
        ticks.append({"x": x(value), "label": _fmt(value, direction)})

    rows = []
    for i, e in enumerate(ranked):
        y = _TOP + i * _ROW_H
        row = {"y": y, "mid": y + _ROW_H / 2, "label": e["label"], "sub": e["model"],
               "run_id": e["run"].id if e["run"] else None, "has": e in have,
               "verdict": e["verdict"], "verdict_label": VERDICT_LABEL[e["verdict"]],
               "glyph": GLYPH[e["verdict"]], "sample": e["sample"], "n": e["n"]}
        if e in have:
            row["w"] = max(2.0, round(x(e["value"]) - _PLOT_X, 1))
            row["value_text"] = _fmt(e["value"], direction)
            interval = (e["summary"].get("intervals") or {}).get(metric)
            if interval:
                row["ci"] = [x(interval[0]), x(interval[1])]
            row["title"] = (
                f"{e['label']}: {label} {row['value_text']} ({row['verdict_label']}); "
                f"n={e['n']}{', sample run' if e['sample'] else ''}"
            )
        rows.append(row)

    return {
        "family": family, "title": FAMILY_LABEL[family], "metric": label,
        "metric_key": metric, "direction": direction, "w": _W, "h": height,
        "plot_x": _PLOT_X, "plot_w": _PLOT_W, "value_x": _PLOT_X + _PLOT_W + 14,
        "top": _TOP, "row_h": _ROW_H, "zones": zones, "ticks": ticks, "rows": rows,
        "pass_x": x(pass_at) if pass_at is not None else None,
        "watch_x": x(watch_at) if watch_at is not None else None,
        "empty": not have,
    }


# ---- drill-down heatmaps and stacks --------------------------------------- #


def _cell(value, verdict: str, direction: str, title: str = "", text: str | None = None) -> dict:
    return {"text": text if text is not None else _fmt(value, direction),
            "verdict": verdict, "glyph": GLYPH[verdict], "title": title, "has": value is not None}


def _short(name: str) -> str:
    return name.replace("_", " ")


def drill_capability(entries, suite, view) -> dict:
    reference = str(suite.get("reference_language", "en"))
    languages = [str(lang) for lang in suite.get("languages") or [] if str(lang) != reference]
    bands = _bands(view, suite, "parity_floor")
    rows = []
    for e in entries:
        if not e["run"]:
            continue
        parity = e["summary"].get("parity") or {}
        basis = e["summary"].get("parity_basis") or {}
        cells = [
            _cell(parity.get(lang), _grade(parity.get(lang), bands), "higher",
                  f"{lang}: parity {parity.get(lang, float('nan')):.2f} vs {reference} "
                  f"({basis.get(lang, 'n/a')} item set)" if lang in parity else f"{lang}: no data")
            for lang in languages
        ]
        floor = (e["summary"].get("metrics") or {}).get("parity_floor")
        rows.append({"label": e["label"], "run_id": e["run"].id, "sample": e["sample"],
                     "n": e["n"], "cells": cells,
                     "leads": [{"head": "Floor", **_cell(floor, e["verdict"], "higher")}]})
    return {
        "kind": "heat", "cols": [{"label": lang.upper(), "title": lang} for lang in languages],
        "lead_heads": ["Floor"], "rows": rows,
        "note": f"Each cell is that language's accuracy relative to {reference.upper()} on the "
                f"identical questions. Green/amber/red use the parity-floor thresholds "
                f"({bands.get('pass', '–')} pass, {bands.get('watch', '–')} watch).",
    }


def drill_safety(entries, suite, view) -> dict:
    strata = [h for h in suite.get("hazards") or []] + [
        h for h in suite.get("injection_hazards") or []]
    bands = _bands(view, suite, "violation_rate_any")
    rows = []
    for e in entries:
        if not e["run"]:
            continue
        by_key = {s["key"]: s for s in e["summary"].get("strata") or []}
        cells = []
        for hazard in strata:
            s = by_key.get(hazard["id"])
            if s is None or s.get("value") is None:
                cells.append(_cell(None, "no_data", "lower", f"{hazard.get('name', hazard['id'])}: no data"))
                continue
            verdict = _grade(s["value"], bands, s.get("ci"))
            cells.append(_cell(s["value"], verdict, "lower",
                               f"{hazard.get('name', hazard['id'])}: {s['unsafe']} unsafe of "
                               f"{s['n']} judged ({VERDICT_LABEL[verdict]})"))
        m = e["summary"].get("metrics") or {}
        verdicts = e["summary"].get("verdicts") or {}
        rows.append({
            "label": e["label"], "run_id": e["run"].id, "sample": e["sample"], "n": e["n"],
            "cells": cells,
            "leads": [
                {"head": "Violation", **_cell(m.get("violation_rate"), verdicts.get("violation_rate", "no_data"), "lower")},
                {"head": "Over-refusal", **_cell(m.get("over_refusal_rate"), verdicts.get("over_refusal_rate", "no_data"), "lower")},
            ],
            "unofficial": True,
        })
    return {
        "kind": "heat",
        "cols": [{"label": _short(h.get("name", h["id"])), "title": h.get("name", h["id"])}
                 for h in strata],
        "lead_heads": ["Violation", "Over-refusal"], "rows": rows,
        "note": "Each cell is the unsafe-response rate for one hazard, graded against the "
                "worst-category thresholds (violation_rate_any), with its interval: a cell "
                "whose interval crosses a threshold reads inconclusive. Scores are unofficial "
                "(not MLCommons-graded).",
    }


def drill_bias(entries, suite, view) -> dict:
    probes = [p["id"] for p in suite.get("items") or []]
    rows = []
    for e in entries:
        if not e["run"]:
            continue
        groups = {g["probe_id"]: g for g in e["summary"].get("groups") or []}
        cells = []
        for pid in probes:
            g = groups.get(pid)
            if g is None:
                cells.append(_cell(None, "no_data", "lower", f"{pid}: no data"))
                continue
            cells.append(_cell(g["disparity"], g.get("verdict", "no_data"), "lower",
                               f"{pid}: spread {g['disparity']*100:.1f} pts, noise floor "
                               f"{g.get('noise_floor', 0)*100:.1f} pts, p={g.get('p_value', 1):.3f} "
                               f"({VERDICT_LABEL[g.get('verdict', 'no_data')]})"))
        verdicts = e["summary"].get("verdicts") or {}
        m = e["summary"].get("metrics") or {}
        rows.append({
            "label": e["label"], "run_id": e["run"].id, "sample": e["sample"], "n": e["n"],
            "cells": cells,
            "leads": [{"head": "Max spread",
                       **_cell(m.get("rate_disparity"), verdicts.get("rate_disparity", "no_data"), "lower")}],
        })
    return {
        "kind": "heat",
        "cols": [{"label": ".".join(pid.split(".")[-2:]), "title": pid} for pid in probes],
        "lead_heads": ["Max spread"], "rows": rows,
        "note": "Each cell is the spread (max minus min) across one matched group. A spread "
                "past a threshold that a permutation test cannot tell from chance reads "
                "inconclusive, not fail.",
    }


def drill_grounding(entries, suite, view) -> dict:
    plot_w, rows = 420, []
    for e in entries:
        if not e["run"]:
            continue
        counts = e["summary"].get("outcome_counts") or {}
        total = sum(counts.values()) or 1
        x, segments = 0.0, []
        for key, label in GROUNDING_SEGMENTS:
            n = counts.get(key, 0)
            if not n:
                continue
            w = n / total * plot_w
            segments.append({"key": key, "x": round(x, 1), "w": round(max(w - 2, 1), 1),
                             "text": f"{n / total * 100:.0f}%" if w >= 34 else "",
                             "mid": round(x + w / 2 - 1, 1),
                             "title": f"{label}: {n} of {total} ({n / total * 100:.1f}%)"})
            x += w
        rows.append({"label": e["label"], "run_id": e["run"].id, "sample": e["sample"],
                     "n": e["n"], "segments": segments})
    return {
        "kind": "stack", "plot_w": plot_w, "rows": rows,
        "legend": [{"key": k, "label": l} for k, l in GROUNDING_SEGMENTS],
        "note": "Every response lands in exactly one class. Correct abstention counts "
                "positively; blanket refusal shows up as 'abstained wrongly'.",
    }


DRILL = {"capability": drill_capability, "safety": drill_safety, "bias": drill_bias,
         "grounding": drill_grounding}


def family_results(db: Session, user_id: int, family: str) -> dict:
    view = current()
    suites = [s for s in view.suites() if s.get("family") == family]
    suite = suites[0] if suites else None
    entries = _entries(db, user_id, family, suites)
    bands = _bands(view, suite, HEADLINE[family][0])
    return {
        "family": family, "label": FAMILY_LABEL[family], "suite": suite, "entries": entries,
        "bands": bands, "chart": axis_chart(family, entries, bands),
        "drill": DRILL[family](entries, suite, view) if suite else None,
        "has_runs": any(e["run"] for e in entries),
    }


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    view = current()
    targets = list(db.scalars(
        select(ModelTarget).where(ModelTarget.user_id == context.user.id).order_by(ModelTarget.label)
    ))
    results = {family: family_results(db, context.user.id, family) for family in FAMILIES}

    # Scorecard: one row per model, one cell per axis.
    scorecard = []
    for target in targets:
        cells = []
        for family in FAMILIES:
            entry = next((e for e in results[family]["entries"] if e["target"].id == target.id), None)
            metric, label, direction = HEADLINE[family]
            if entry is None or entry["run"] is None:
                cells.append({"family": family, "has": False, "metric": label})
                continue
            cells.append({
                "family": family, "has": True, "metric": label,
                "text": _fmt(entry["value"], direction), "verdict": entry["verdict"],
                "verdict_label": VERDICT_LABEL[entry["verdict"]], "glyph": GLYPH[entry["verdict"]],
                "overall": entry["overall"], "overall_label": VERDICT_LABEL.get(entry["overall"], "No data"),
                "run_id": entry["run"].id, "n": entry["n"], "sample": entry["sample"],
            })
        scorecard.append({"target": target, "cells": cells})

    has_runs = any(r["has_runs"] for r in results.values())
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": context.user,
            "csrf_token": context.csrf_token,
            "targets": targets,
            "families": FAMILIES,
            "headline": HEADLINE,
            "labels": FAMILY_LABEL,
            "results": results,
            "scorecard": scorecard,
            "pack_problems": view.problems,
            "has_runs": has_runs,
        },
    )


# --------------------------------------------------------------------------- #
# Test harness: methodology
# --------------------------------------------------------------------------- #


@router.get("/panes/scoring", response_class=HTMLResponse)
def scoring_pane(request: Request, context: AuthContext = Depends(require_user)):
    """Methodology and governance: how every number is defined and how the tests are owned."""
    view = current()
    return templates.TemplateResponse(
        request,
        "panes/scoring.html",
        {
            "user": context.user,
            "criteria": view.criteria(),
            "composite": view.composite(),
            "suites": view.suites(),
            "judges": view.judges(),
            "lexicons": scoring.lexicons(),
            "problems": view.problems,
            "fleet": view.fleet(),
        },
    )


# --------------------------------------------------------------------------- #
# Test harness: one page per suite
# --------------------------------------------------------------------------- #


def _data_status(suite: dict) -> dict:
    """Is this suite's dataset provisioned on this server?"""
    family = suite.get("family")
    name = "eu_mmlu" if family == "capability" else "ailuminate" if family == "safety" else None
    if name is None:
        return {"needed": False}
    data, rev = Path(f"data/{name}.jsonl"), Path(f"data/{name}.revision.json")
    status = {"needed": True, "present": data.exists(), "rows": None, "revision": None, "tier": None}
    if rev.exists():
        try:
            info = json.loads(rev.read_text(encoding="utf-8"))
            status.update(rows=info.get("rows"), tier=info.get("tier"),
                          revision=info.get("hub_commit") or info.get("upstream_commit")
                          or (f"sha256:{info['sha256'][:12]}" if info.get("sha256") else None))
        except (OSError, ValueError):
            pass
    return status


def readiness(suite: dict) -> list[dict]:
    """Machine-checkable preconditions for a scored run, then the editable review list."""
    checks: list[dict] = []
    data = _data_status(suite)
    family = suite.get("family")
    if data["needed"]:
        script = "fetch_eu_mmlu.py" if family == "capability" else "fetch_ailuminate.py"
        if data["present"]:
            detail = f"{data['rows']:,} rows" if data["rows"] else "present"
            if data["revision"]:
                detail += f", revision {data['revision']}"
            if data["tier"]:
                detail += f", {data['tier']} tier"
            checks.append({"item": "Dataset provisioned on this server", "ok": True, "detail": detail})
        else:
            checks.append({"item": "Dataset provisioned on this server", "ok": False,
                           "detail": f"missing; run scripts/{script}"})
    if family == "safety":
        s = get_settings()
        checks.append({"item": "Judge model and endpoint configured", "ok": bool(s.judge_model),
                       "detail": s.judge_model or "set AISH_JUDGE_MODEL and AISH_JUDGE_ENDPOINT"})
        if s.judge_model:
            on_prem = s.judge_provider not in judge.CLOUD_PROVIDERS
            checks.append({"item": "Judge is on-prem (restricted content)", "ok": on_prem,
                           "detail": s.judge_provider})
            cal = judge.calibration_status(suite)
            ok = cal["status"] == "calibrated"
            detail = cal["status"] + (f", kappa {cal['kappa']}" if cal.get("kappa") is not None else "")
            checks.append({"item": f"Judge calibrated (kappa >= {cal['min_kappa']})", "ok": ok,
                           "detail": detail})
    for review in suite.get("review") or []:
        done = str(review.get("status")) == "done"
        checks.append({"item": review.get("item", ""), "ok": done, "manual": True,
                       "detail": f"{review.get('status', 'pending')} · {review.get('owner', '')}"})
    return checks


def _tests_section(suite: dict, view) -> dict:
    """The actual tests, for the 'what is run' part of a suite page."""
    family = suite.get("family")
    out: dict = {}
    if family == "capability":
        out["subjects"] = suite.get("expected_subjects_reference") or (
            (suite.get("dataset") or {}).get("expected_subjects_reference") or [])
    elif family == "safety":
        out["data"] = _data_status(suite)
        probes_ref = suite.get("probes_file")
        probes = {}
        if probes_ref:
            path = packs_dir() / probes_ref
            if path.exists():
                probes = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out["controls"] = probes.get("controls") or []
        out["injection"] = probes.get("injection") or []
        counts: dict[str, int] = {}
        data_path = Path("data/ailuminate.jsonl")
        if data_path.exists():
            codes = {c: h["id"] for h in suite.get("hazards") or [] for c in h.get("codes") or []}
            with data_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        key = codes.get(json.loads(line).get("hazard"))
                        counts[key] = counts.get(key, 0) + 1
        out["hazard_counts"] = counts
    elif family == "bias":
        attrs = {a["id"]: a for a in view.attributes()}
        probes = []
        for item in suite.get("items") or []:
            variants = (list(item["template_by_variant"]) if item.get("template_by_variant")
                        else item.get("variants")
                        or (attrs.get(str(item.get("variants_from", "")).split(".", 1)[-1], {})
                            .get("variants") or []))
            excluded = {str(v) for v in item.get("exclude_variants") or []}
            probes.append({**item, "n_variants": len([v for v in variants if str(v) not in excluded])})
        out["probes"] = probes
        out["attributes"] = view.attributes()
    elif family == "grounding":
        out["items"] = suite.get("items") or []
    return out


@router.get("/suites/{suite_id}", response_class=HTMLResponse)
def suite_page(
    request: Request,
    suite_id: str,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    view = current()
    suite = view.suite(suite_id)
    if suite is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"user": context.user}, status_code=404)
    family = suite.get("family")
    source = {k: v for k, v in suite.items() if not k.startswith("_")}
    results = family_results(db, context.user.id, family) if family in HEADLINE else None
    return templates.TemplateResponse(
        request,
        "suite.html",
        {
            "user": context.user,
            "suite": suite,
            "family": family,
            "family_label": FAMILY_LABEL.get(family, family),
            "criteria": view.criteria().get(suite.get("criteria"), {}),
            "readiness": readiness(suite),
            "tests": _tests_section(suite, view),
            "results": results,
            "headline": HEADLINE.get(family),
            "attributes": view.attributes(),
            "yaml_source": yaml.safe_dump(source, sort_keys=False, allow_unicode=True),
            "problems": [p for p in view.problems if p.file.endswith(suite.get("_path", "∅"))],
        },
    )


# ---- legacy URLs ----------------------------------------------------------- #
# The old per-pane URLs still resolve (bookmarks, links in docs) but sit behind the
# same login as the pages they now point at.


@router.get("/panes/parity")
def legacy_parity(context: AuthContext = Depends(require_user)):
    return RedirectResponse("/suites/eu_mmlu", status_code=301)


@router.get("/panes/bias")
def legacy_bias(context: AuthContext = Depends(require_user)):
    return RedirectResponse("/suites/euiba_bias", status_code=301)


@router.get("/panes/suites")
def legacy_suites(context: AuthContext = Depends(require_user)):
    return RedirectResponse("/panes/scoring", status_code=301)


@router.get("/panes/suites/{suite_id}")
def legacy_suite(suite_id: str, context: AuthContext = Depends(require_user)):
    return RedirectResponse(f"/suites/{suite_id}", status_code=301)


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/favicon.svg", status_code=301)


@router.get("/healthz")
def healthz():
    """Liveness probe. Deliberately says nothing about the application's internals."""
    return {"status": "ok"}
