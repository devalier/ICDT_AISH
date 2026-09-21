"""The GUI panes: comparison, language parity, bias basis, scoring basis, suites."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import ModelTarget, Run
from ..packs import current
from .deps import AuthContext, optional_user, require_user, templates

router = APIRouter()

FAMILIES = ("capability", "safety", "bias", "grounding")

# The headline metric shown per family on the comparison pane. Full metric sets are
# on the run page; this is the one number each family leads with.
HEADLINE = {
    "capability": ("parity_floor", "Parity floor", "higher"),
    "safety": ("violation_rate", "Violation rate", "lower"),
    "bias": ("rate_disparity", "Max disparity", "lower"),
    "grounding": ("hallucination_rate", "Hallucination rate", "lower"),
}


@router.get("/")
def root(context: AuthContext | None = Depends(optional_user)):
    return RedirectResponse("/dashboard" if context else "/login", status_code=303)


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


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    view = current()
    targets = list(
        db.scalars(
            select(ModelTarget)
            .where(ModelTarget.user_id == context.user.id)
            .order_by(ModelTarget.label)
        )
    )
    latest = _latest_runs(db, context.user.id)
    suites_by_family = {family: [s for s in view.suites() if s.get("family") == family]
                        for family in FAMILIES}

    grid = []
    for target in targets:
        row = {"target": target, "cells": []}
        for family in FAMILIES:
            cell = {"family": family, "run": None, "value": None, "verdict": "no_data",
                    "metric": HEADLINE[family][1], "direction": HEADLINE[family][2]}
            for suite in suites_by_family[family]:
                run = latest.get((target.id, suite.get("id")))
                if run is None:
                    continue
                summary = json.loads(run.summary_json or "{}")
                metric_key = HEADLINE[family][0]
                cell["run"] = run
                cell["value"] = (summary.get("metrics") or {}).get(metric_key)
                cell["verdict"] = summary.get("verdict", "no_data")
                cell["sample_mode"] = summary.get("sample_mode", True)
                cell["n"] = summary.get("n", 0)
                break
            row["cells"].append(cell)
        grid.append(row)

    # The comparison bars claim to show the watch and fail marks, so the page has to
    # carry them rather than assert them in prose.
    thresholds = {}
    for family in FAMILIES:
        metric_key = HEADLINE[family][0]
        for suite in suites_by_family[family]:
            bands = (view.criteria().get(suite.get("criteria"), {}).get("metrics") or {})
            if metric_key in bands:
                thresholds[family] = bands[metric_key]
                break

    run_count = db.scalar(
        select(Run).where(Run.user_id == context.user.id).order_by(Run.id.desc()).limit(1)
    )

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": context.user,
            "csrf_token": context.csrf_token,
            "grid": grid,
            "families": FAMILIES,
            "headline": HEADLINE,
            "targets": targets,
            "thresholds": thresholds,
            "pack_problems": view.problems,
            "has_runs": run_count is not None,
        },
    )


@router.get("/panes/parity", response_class=HTMLResponse)
def parity_pane(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    view = current()
    suite = view.suite("eu_mmlu") or {}
    languages = [str(lang) for lang in suite.get("languages") or []]
    reference = str(suite.get("reference_language", "en"))

    latest = _latest_runs(db, context.user.id)
    labels = {
        t.id: t.label
        for t in db.scalars(select(ModelTarget).where(ModelTarget.user_id == context.user.id))
    }

    rows = []
    for (target_id, suite_id), run in latest.items():
        if suite_id != "eu_mmlu":
            continue
        summary = json.loads(run.summary_json or "{}")
        accuracy = summary.get("language_accuracy") or {}
        parity = summary.get("parity") or {}
        rows.append(
            {
                "label": labels.get(target_id, f"target {target_id}"),
                "run": run,
                "accuracy": accuracy,
                "parity": parity,
                "floor": (summary.get("metrics") or {}).get("parity_floor"),
                "sample_mode": summary.get("sample_mode", True),
                "n": summary.get("n", 0),
            }
        )

    return templates.TemplateResponse(
        request,
        "panes/parity.html",
        {
            "user": context.user,
            "suite": suite,
            "languages": languages,
            "reference": reference,
            "rows": sorted(rows, key=lambda r: r["label"]),
        },
    )


@router.get("/panes/bias", response_class=HTMLResponse)
def bias_pane(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    """The basis pane: what grounds a bias finding, shown before any number."""
    view = current()
    suite = view.suite("euiba_bias") or {}
    criteria = view.criteria().get(suite.get("criteria"), {})

    latest = _latest_runs(db, context.user.id)
    labels = {
        t.id: t.label
        for t in db.scalars(select(ModelTarget).where(ModelTarget.user_id == context.user.id))
    }
    findings = []
    for (target_id, suite_id), run in latest.items():
        if suite_id != "euiba_bias":
            continue
        summary = json.loads(run.summary_json or "{}")
        findings.append(
            {
                "label": labels.get(target_id, f"target {target_id}"),
                "run": run,
                "groups": summary.get("groups") or [],
                "verdicts": summary.get("verdicts") or {},
                "verdict": summary.get("verdict", "no_data"),
                "resamples": summary.get("bootstrap_resamples"),
                "capped_from": summary.get("bootstrap_capped_from"),
            }
        )

    return templates.TemplateResponse(
        request,
        "panes/bias.html",
        {
            "user": context.user,
            "suite": suite,
            "attributes": view.attributes(),
            "criteria": criteria,
            "findings": sorted(findings, key=lambda f: f["label"]),
        },
    )


@router.get("/panes/scoring", response_class=HTMLResponse)
def scoring_pane(request: Request, context: AuthContext = Depends(require_user)):
    """The scoring pane: every metric, threshold, weight and lexicon on one page."""
    from ..scoring import lexicons

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
            "lexicons": lexicons(),
        },
    )


@router.get("/panes/suites", response_class=HTMLResponse)
def suites_pane(request: Request, context: AuthContext = Depends(require_user)):
    view = current()
    return templates.TemplateResponse(
        request,
        "panes/suites.html",
        {
            "user": context.user,
            "suites": view.suites(),
            "problems": view.problems,
            "fleet": view.fleet(),
        },
    )


@router.get("/panes/suites/{suite_id}", response_class=HTMLResponse)
def suite_detail(request: Request, suite_id: str, context: AuthContext = Depends(require_user)):
    import yaml

    view = current()
    suite = view.suite(suite_id)
    if suite is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"user": context.user}, status_code=404
        )
    source = {k: v for k, v in suite.items() if not k.startswith("_")}
    return templates.TemplateResponse(
        request,
        "panes/suite_detail.html",
        {
            "user": context.user,
            "suite": suite,
            "criteria": view.criteria().get(suite.get("criteria"), {}),
            "yaml_source": yaml.safe_dump(source, sort_keys=False, allow_unicode=True),
            "problems": [p for p in view.problems if p.file.endswith(suite.get("_path", "∅"))],
        },
    )


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/favicon.svg", status_code=301)


@router.get("/healthz")
def healthz():
    """Liveness probe. Deliberately says nothing about the application's internals."""
    return {"status": "ok"}
