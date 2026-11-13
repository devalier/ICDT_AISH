"""Starting runs, watching them, and reading the evidence behind a number."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import custom, judge, prefs
from ..config import get_settings
from ..db import get_session
from ..models import ModelTarget, Result, Run
from ..packs import current
from ..runner import RunError, build_items, dataset_revision, queue_run, run_temperature
from .deps import AuthContext, audit, require_user, templates, verify_csrf

router = APIRouter()

DEFAULT_SEED = 20260921
# Parallel runs are capped: each opens its own concurrent requests to the model provider.
MAX_PARALLEL_MODELS = 3


def _owned_run(db: Session, context: AuthContext, run_id: int) -> Run | None:
    return db.scalar(select(Run).where(Run.id == run_id, Run.user_id == context.user.id))


def _started_runs(db: Session, context: AuthContext, started) -> list[dict]:
    """Runs a user has just launched (ids in the query string), limited to their own."""
    if not isinstance(started, str) or not started:
        return []
    ids = [int(x) for x in started.split(",") if x.strip().isdigit()][:MAX_PARALLEL_MODELS]
    out = []
    for run_id in ids:
        run = _owned_run(db, context, run_id)
        if run is not None:
            target = db.get(ModelTarget, run.target_id)
            out.append({"id": run.id, "label": target.label if target else f"run {run.id}"})
    return out


@router.get("/runs", response_class=HTMLResponse)
def list_runs(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
    error: str | None = None,
    suite: str | None = None,
    model: list[int] = Query(default=[]),
    started: str | None = None,
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
            "selected_suite": suite,
            "selected_models": {m for m in (model if isinstance(model, list) else []) if isinstance(m, int)},
            "cloud_providers": judge.CLOUD_PROVIDERS,
            "safety_cloud_allowed": prefs.cloud_allowed(db, context.user.id),
            "safety_toggle_available": prefs.toggle_available(),
            "safety_switch_name": prefs.SWITCH_NAME,
            "safety_warning": prefs.WARNING,
            "restricted_suite_ids": [s["id"] for s in view.suites() if s.get("restricted")],
            "started": _started_runs(db, context, started),
            "suite_names": {s["id"]: f"{s.get('menu_label', s.get('family', ''))} · {s.get('title', s['id'])}"
                            for s in view.suites()},
            "my_versions": {s["id"]: row for s in view.suites()
                            if (row := custom.active(db, context.user.id, s["id"])) is not None},
            "run_fingerprints": {r.id: json.loads(r.manifest_json or "{}").get("fingerprint") for r in runs},
        },
    )


@router.post("/runs", dependencies=[Depends(verify_csrf)])
def start_run(
    request: Request,
    suite_id: str = Form(...),
    target_ids: list[int] = Form(default=[]),
    target_id: int | None = Form(None),
    mode: str = Form("sample"),
    confirm_cloud_safety: str = Form(""),
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    """Start one run per selected model (up to MAX_PARALLEL_MODELS), all on the same prompts
    and settings so their results are directly comparable. Everything is validated before any
    run is created, so a refusal never leaves half a comparison behind."""
    view = current()
    # "euiba_bias" is the official suite; "euiba_bias:custom" is the user's own version of it.
    raw_suite_choice = suite_id
    use_custom = suite_id.endswith(":custom")
    suite_id = suite_id.removesuffix(":custom")
    suite = view.suite(suite_id)
    chosen = list(dict.fromkeys([*target_ids, *([target_id] if target_id is not None else [])]))

    def fail(message: str, status_code: int = 400):
        response = list_runs(request, db, context, error=message, suite=raw_suite_choice, model=chosen,
                             started=None)
        response.status_code = status_code
        return response

    if suite is None:
        return fail("That suite is not defined.")
    custom_row = None
    if use_custom:
        custom_row = custom.active(db, context.user.id, suite_id)
        if custom_row is None:
            return fail("You have no customised version of that suite. Customise it first, or run the official one.")
        checked, problems = custom.check(suite, custom_row.yaml_text, view)
        if checked is None:
            first = next((p for p in problems if p.severity == "error"), None)
            return fail("Your saved version no longer passes validation: "
                        f"{first.message if first else 'see the editor'}.")
        checked["_custom"] = {"id": custom_row.id, "fingerprint": custom_row.fingerprint}
        suite = checked
    if not chosen:
        return fail("Select at least one model to test.")
    if len(chosen) > MAX_PARALLEL_MODELS:
        return fail(f"Select at most {MAX_PARALLEL_MODELS} models for one comparison; "
                    f"you selected {len(chosen)}.")
    targets = list(db.scalars(
        select(ModelTarget).where(ModelTarget.id.in_(chosen), ModelTarget.user_id == context.user.id)
    ))
    if len(targets) != len(chosen):
        return fail("You can only select your own models.")
    targets.sort(key=lambda t: chosen.index(t.id))

    sample = mode != "full"
    cloud_ok = False
    if suite.get("restricted"):
        cloud = [t.label for t in targets if t.provider in judge.CLOUD_PROVIDERS]
        if cloud:
            names = ", ".join(cloud)
            if not prefs.toggle_available():
                return fail(f"The Safety harness cannot run on public cloud models ({names}) on this "
                            "deployment: the administrator has disabled it. Choose on-prem models.", 409)
            if not prefs.cloud_allowed(db, context.user.id):
                return fail(f"The Safety harness is not allowed on public cloud models ({names}). "
                            f"To allow it, switch on \u201c{prefs.SWITCH_NAME}\u201d on the Safety "
                            "test suite page (Test suites \u2192 Safety), then come back. "
                            "On-prem models can run it without this.", 409)
            if not confirm_cloud_safety:
                return fail(f"You selected public cloud models ({names}) for the Safety harness. "
                            "Tick the box confirming you understand the risk to your provider "
                            "account, then run again.", 409)
            cloud_ok = True
    try:
        if suite.get("family") == "safety":
            judge.require_configured(restricted=bool(suite.get("restricted")))
        items = build_items(suite, view, sample=sample, seed=DEFAULT_SEED)
        if suite.get("family") == "bias" and any(i.meta.get("judge_scoring") for i in items):
            judge.require_configured(restricted=bool(suite.get("restricted")))
    except RunError as exc:
        return fail(str(exc), 409)

    limit = get_settings().max_items_per_run
    if len(items) > limit:
        return fail(
            f"That run would send {len(items):,} prompts, above this instance's limit of "
            f"{limit:,}. Use sample mode, or ask an administrator to raise "
            "AISH_MAX_ITEMS_PER_RUN. The harness will not shorten a run silently, "
            "because dropping prompts would distort the categories the score is averaged over.",
            409,
        )

    run_ids = []
    for target in targets:
        run = Run(
            user_id=context.user.id,
            target_id=target.id,
            suite_id=suite_id,
            suite_version=(custom.label(str(suite.get("version", "")), custom_row.fingerprint)
                           if custom_row else str(suite.get("version", ""))),
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
                    "dataset_revision": dataset_revision(suite),
                    "provider": target.provider,
                    "model": target.model_name,
                    "sample_mode": sample,
                    "seed": DEFAULT_SEED,
                    "temperature": run_temperature(suite, sample),
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    **({"cloud_safety_confirmed": True}
                       if cloud_ok and target.provider in judge.CLOUD_PROVIDERS else {}),
                    **({"custom_suite_id": custom_row.id, "fingerprint": custom_row.fingerprint}
                       if custom_row else {}),
                    **({"judge": judge.describe()} if suite.get("family") == "safety" else {}),
                }
            ),
        )
        db.add(run)
        db.flush()
        run_ids.append(run.id)
        audit(db, "run.started", request=request, user_id=context.user.id,
              detail=f"{suite_id}{' (custom)' if custom_row else ''} -> {target.label}")
    db.commit()

    failed = []
    for run_id in run_ids:
        try:
            queue_run(run_id)
        except RunError as exc:
            run = db.get(Run, run_id)
            if run is not None:
                run.status = "failed"
                run.error = str(exc)[:600]
            failed.append(str(exc))
    if failed and len(failed) == len(run_ids):
        db.commit()
        return fail(failed[0], 503)
    db.commit()

    if len(run_ids) == 1:
        return RedirectResponse(f"/runs/{run_ids[0]}", status_code=303)
    return RedirectResponse("/runs?started=" + ",".join(str(r) for r in run_ids), status_code=303)


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
    custom_fingerprint = manifest.get("fingerprint")
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
            "custom_fingerprint": custom_fingerprint,
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
