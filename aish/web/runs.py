"""Starting runs, watching them, and reading the evidence behind a number."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import ModelTarget, Result, Run
from ..packs import current
from ..runner import RunError, build_items, queue_run
from .deps import AuthContext, audit, require_user, templates, verify_csrf

router = APIRouter()

DEFAULT_SEED = 20260921


def _owned_run(db: Session, context: AuthContext, run_id: int) -> Run | None:
    return db.scalar(select(Run).where(Run.id == run_id, Run.user_id == context.user.id))


@router.get("/runs", response_class=HTMLResponse)
def list_runs(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
    error: str | None = None,
):
    view = current()
    runs = list(
        db.scalars(
            select(Run)
            .where(Run.user_id == context.user.id)
            .order_by(Run.created_at.desc())
            .limit(100)
        )
    )
    targets = list(
        db.scalars(
            select(ModelTarget).where(
                ModelTarget.user_id == context.user.id, ModelTarget.enabled.is_(True)
            )
        )
    )
    return templates.TemplateResponse(
        request,
        "runs.html",
        {
            "user": context.user,
            "csrf_token": context.csrf_token,
            "runs": runs,
            "targets": targets,
            "suites": view.suites(),
            "summaries": {run.id: json.loads(run.summary_json or "{}") for run in runs},
            "labels": {t.id: t.label for t in db.scalars(
                select(ModelTarget).where(ModelTarget.user_id == context.user.id)
            )},
            "error": error,
        },
    )


@router.post("/runs", dependencies=[Depends(verify_csrf)])
def start_run(
    request: Request,
    suite_id: str = Form(...),
    target_id: int = Form(...),
    mode: str = Form("sample"),
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    view = current()
    suite = view.suite(suite_id)
    target = db.scalar(
        select(ModelTarget).where(
            ModelTarget.id == target_id, ModelTarget.user_id == context.user.id
        )
    )

    def fail(message: str, status_code: int = 400):
        response = list_runs(request, db, context, error=message)
        response.status_code = status_code
        return response

    if suite is None:
        return fail("That suite is not defined.")
    if target is None:
        return fail("Select one of your own model targets.")

    sample = mode != "full"
    try:
        items = build_items(suite, view, sample=sample, seed=DEFAULT_SEED)
    except RunError as exc:
        return fail(str(exc), 409)

    limit = get_settings().max_items_per_run
    if len(items) > limit:
        return fail(
            f"That run would send {len(items):,} items, above this instance's limit of "
            f"{limit:,}. Use sample mode, or ask an administrator to raise "
            "AISH_MAX_ITEMS_PER_RUN. The harness will not shorten a run silently, "
            "because dropping items would distort the strata the score is averaged over.",
            409,
        )

    run = Run(
        user_id=context.user.id,
        target_id=target.id,
        suite_id=suite_id,
        suite_version=str(suite.get("version", "")),
        family=str(suite.get("family", "")),
        status="queued",
        sample_mode=sample,
        items_total=len(items),
        manifest_json=json.dumps(
            {
                "suite_id": suite_id,
                "suite_version": suite.get("version"),
                "scorer": suite.get("scorer"),
                "criteria": suite.get("criteria"),
                "dataset_revision": (suite.get("dataset") or {}).get("revision"),
                "provider": target.provider,
                "model": target.model_name,
                "sample_mode": sample,
                "seed": DEFAULT_SEED,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1024,
            }
        ),
    )
    db.add(run)
    db.flush()
    run_id = run.id
    audit(db, "run.started", request=request, user_id=context.user.id,
          detail=f"{suite_id} -> {target.label}")
    db.commit()

    try:
        queue_run(run_id)
    except RunError as exc:
        with_failure = db.get(Run, run_id)
        if with_failure is not None:
            with_failure.status = "failed"
            with_failure.error = str(exc)[:600]
        return fail(str(exc), 503)

    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(
    request: Request,
    run_id: int,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    run = _owned_run(db, context, run_id)
    if run is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"user": context.user}, status_code=404
        )

    view = current()
    suite = view.suite(run.suite_id) or {}
    summary = json.loads(run.summary_json or "{}")
    manifest = json.loads(run.manifest_json or "{}")
    target = db.get(ModelTarget, run.target_id)

    outcome_counts = dict(
        db.execute(
            select(Result.outcome, func.count())
            .where(Result.run_id == run.id)
            .group_by(Result.outcome)
        ).all()
    )
    failures = list(
        db.scalars(
            select(Result)
            .where(Result.run_id == run.id, Result.error != "")
            .limit(5)
        )
    )

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "user": context.user,
            "csrf_token": context.csrf_token,
            "run": run,
            "suite": suite,
            "target": target,
            "summary": summary,
            "manifest": manifest,
            "criteria": view.criteria().get(suite.get("criteria"), {}),
            "outcome_counts": outcome_counts,
            "failures": failures,
        },
    )


@router.get("/runs/{run_id}/status")
def run_status(
    run_id: int,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    """Polled by the run page so progress advances without a manual refresh."""
    run = _owned_run(db, context, run_id)
    if run is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(
        {
            "status": run.status,
            "items_done": run.items_done,
            "items_total": run.items_total,
            "error": run.error,
        }
    )


@router.get("/runs/{run_id}/evidence", response_class=HTMLResponse)
def run_evidence(
    request: Request,
    run_id: int,
    page: int = Query(1, ge=1, le=1000),
    outcome: str = Query("", max_length=40),
    stratum: str = Query("", max_length=120),
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    """Every reported number is one click from the prompts and responses behind it."""
    run = _owned_run(db, context, run_id)
    if run is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"user": context.user}, status_code=404
        )

    per_page = 25
    query = select(Result).where(Result.run_id == run.id)
    count_query = select(func.count()).select_from(Result).where(Result.run_id == run.id)
    if outcome:
        query = query.where(Result.outcome == outcome)
        count_query = count_query.where(Result.outcome == outcome)
    if stratum:
        query = query.where(Result.stratum == stratum)
        count_query = count_query.where(Result.stratum == stratum)

    total = db.scalar(count_query) or 0
    rows = list(
        db.scalars(query.order_by(Result.id).offset((page - 1) * per_page).limit(per_page))
    )
    outcomes = [
        value for (value,) in db.execute(
            select(Result.outcome).where(Result.run_id == run.id).distinct()
        ).all()
    ]

    return templates.TemplateResponse(
        request,
        "evidence.html",
        {
            "user": context.user,
            "run": run,
            "rows": rows,
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": max(1, (total + per_page - 1) // per_page),
            "outcome": outcome,
            "stratum": stratum,
            "outcomes": sorted(o for o in outcomes if o),
        },
    )


@router.post("/runs/{run_id}/delete", dependencies=[Depends(verify_csrf)])
def delete_run(
    request: Request,
    run_id: int,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    run = _owned_run(db, context, run_id)
    if run is not None:
        db.delete(run)
        audit(db, "run.deleted", request=request, user_id=context.user.id, detail=str(run_id))
    return RedirectResponse("/runs", status_code=303)
