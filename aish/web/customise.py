"""Customising a test suite: a personal copy of one of the four official suites.

A user edits the suite's YAML in the browser, checks it, previews the prompts it would send,
tries it on one of their own models, and saves it as their version. The official suite is
never touched. Runs of a personal version are labelled as such and compared only with each
other (see aish/custom.py for what a personal version may and may not change).
"""

from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import custom, editor, judge
from ..config import get_settings
from ..db import get_session
from ..models import ModelTarget
from ..packs import current
from harness.loader import packs_dir
from ..providers import ProviderError, get_adapter
from ..providers.base import GenerationParams
from ..runner import (DEFAULT_SEED_FOR_PREVIEW, Item, RunError, build_items, run_temperature,
                      safety_probe_items, score_response)
from ..security import decrypt_secret, redact
from .deps import AuthContext, audit, require_user, templates, verify_csrf

router = APIRouter()

def _suite_or_404(request: Request, suite_id: str, context: AuthContext):
    suite = current().suite(suite_id)
    if suite is None:
        return None, templates.TemplateResponse(
            request, "not_found.html", {"user": context.user}, status_code=404)
    return suite, None


def _preview(suite: dict, view) -> dict:
    """What the suite would send: counts, and a handful of example prompts. Hazardous dataset
    prompts are never shown, so the safety preview lists only the EUIBA probes."""
    if suite.get("family") == "safety":
        items = safety_probe_items(suite)
        note = "Only the EUIBA probes are shown; the MLCommons prompts are restricted content."
    else:
        items = build_items(suite, view, sample=True, seed=DEFAULT_SEED_FOR_PREVIEW)
        note = ""
    by = Counter((i.stratum, i.variant) for i in items)
    examples, seen = [], set()
    for item in items:
        key = item.stratum
        if key in seen or len(examples) >= 8:
            continue
        seen.add(key)
        examples.append({"stratum": item.stratum, "variant": item.variant,
                         "system": item.system, "prompt": item.prompt[:1500],
                         "name": item.meta.get("name_used"),
                         "judged": bool(item.meta.get("judge_scoring"))})
    return {"total": len(items), "groups": len(by), "examples": examples, "note": note,
            "uses_judge": any(i.meta.get("judge_scoring") for i in items)}


async def _try_on_model(suite: dict, view, target: ModelTarget, user_id: int) -> list[dict]:
    """Send up to six of the suite's own prompts to one model and score them, so an author can
    see how a rubric or a prompt actually behaves before spending a run on it."""
    family = suite.get("family")
    items = safety_probe_items(suite) if family == "safety" else build_items(
        suite, view, sample=True, seed=DEFAULT_SEED_FOR_PREVIEW)
    picked, seen = [], set()
    for item in items:
        if item.stratum not in seen and len(picked) < 6:
            seen.add(item.stratum)
            picked.append(item)
    settings = get_settings()
    api_key = (decrypt_secret(target.api_key_ciphertext, f"user:{user_id}")
               if target.api_key_ciphertext else None)
    adapter = get_adapter(target.provider)
    params = GenerationParams(temperature=run_temperature(suite, True), max_tokens=512, seed=1)

    async def one(item: Item) -> dict:
        row = {"stratum": item.stratum, "variant": item.variant, "prompt": item.prompt[:400],
               "response": "", "outcome": "", "error": ""}
        try:
            completion = await adapter.generate(
                model=target.model_name, prompt=item.prompt, system=item.system, api_key=api_key,
                endpoint=target.endpoint or None, params=params,
                timeout=min(settings.request_timeout_seconds, 60.0))
            row["response"] = completion.text[:900]
            row["outcome"], _ = await score_response(family, item, completion.text, completion.refused)
        except (ProviderError, RunError) as exc:
            row["error"] = redact(str(exc), api_key or "")[:300]
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not hide the others
            row["error"] = redact(f"{type(exc).__name__}: {exc}", api_key or "")[:300]
        return row

    return list(await asyncio.gather(*(one(i) for i in picked)))


def _editor_context(view, base: dict) -> editor.Context:
    sources_dir = packs_dir() / "sources"
    return editor.Context(
        attributes=view.attributes(),
        sources=sorted(f"sources/{p.name}" for p in sources_dir.glob("*.md")) if sources_dir.is_dir() else [],
        hazards=[(h["id"], h.get("name", h["id"])) for h in base.get("hazards") or []],
        injection_hazards=[(h["id"], h.get("name", h["id"])) for h in base.get("injection_hazards") or []],
        languages=[str(x) for x in base.get("languages") or []],
    )


def _page(request, db, context, base, *, view, overrides=None, preview=None, tried=None, notice=None,
          error=None, chosen_target=None, open_item=None):
    """Render the structured editor for the user's working copy of ``base``."""
    user = context.user
    text = custom.working_text(db, user.id, base)
    ctx = _editor_context(view, base)
    secs = editor.sections(base["family"], base, ctx)
    suite, parse_error = None, None
    try:
        suite = custom.parse_yaml(text)
    except custom.CustomError as exc:
        parse_error = str(exc)
    checked, problems = custom.check(base, text, view)
    grouped, general = editor.problems_by_item(problems)
    blocks = editor.read_all(suite, secs) if suite else []
    for block in blocks:
        for item in block["items"]:
            key = (block["section"].key, item["index"])
            item["problems"] = grouped.get(key, [])
            if overrides and key in overrides:
                item["values"] = {**item["values"], **overrides[key]}
            item["open"] = bool(item["problems"]) or (open_item == key) or (overrides and key in overrides)
    errors_total = [p for p in problems if p.severity == "error"]
    return templates.TemplateResponse(request, "customise.html", {
        "user": user, "csrf_token": context.csrf_token, "suite": base, "active": "suite:" + base["id"],
        "family_label": base.get("menu_label", base.get("family")),
        "blocks": blocks, "general_problems": general, "parse_error": parse_error,
        "valid": checked is not None, "error_count": len(errors_total),
        "unsaved": custom.has_unsaved_changes(db, user.id, base),
        "custom_row": custom.active(db, user.id, base["id"]),
        "targets": list(db.scalars(select(ModelTarget).where(ModelTarget.user_id == user.id)
                                   .order_by(ModelTarget.label))),
        "chosen_target": chosen_target, "preview": preview, "tried": tried,
        "notice": notice, "error": error,
    })


@router.get("/suites/{suite_id}/customise", response_class=HTMLResponse)
def editor_page(request: Request, suite_id: str, open: str | None = None,
                db: Session = Depends(get_session), context: AuthContext = Depends(require_user)):
    suite, missing = _suite_or_404(request, suite_id, context)
    if missing:
        return missing
    key = None
    if open and "-" in open and open.rsplit("-", 1)[1].isdigit():
        key = (open.rsplit("-", 1)[0], int(open.rsplit("-", 1)[1]))
    return _page(request, db, context, suite, view=current(), open_item=key)


@router.post("/suites/{suite_id}/customise/item", response_class=HTMLResponse,
             dependencies=[Depends(verify_csrf)])
async def edit_item(request: Request, suite_id: str, db: Session = Depends(get_session),
                    context: AuthContext = Depends(require_user)):
    """Save, duplicate, delete or add one test. The draft is updated on every change."""
    base, missing = _suite_or_404(request, suite_id, context)
    if missing:
        return missing
    view = current()
    form = await request.form()
    section_key, op = str(form.get("section", "")), str(form.get("op", "save"))
    try:
        index = int(form.get("index", -1))
    except ValueError:
        index = -1
    secs = editor.sections(base["family"], base, _editor_context(view, base))
    sec = next((s for s in secs if s.key == section_key), None)
    values = editor.form_values(form, sec.fields) if sec else {}
    text = custom.working_text(db, context.user.id, base)
    new_text, errors, focus = editor.apply(text, secs, section_key, op, index, values)
    if new_text is None:
        return _page(request, db, context, base, view=view, error=" ".join(errors),
                     overrides={(section_key, index): values} if op == "save" else None,
                     open_item=(section_key, index))
    try:
        custom.put_draft(db, context.user.id, suite_id, new_text)
    except custom.CustomError as exc:
        return _page(request, db, context, base, view=view, error=str(exc))
    audit(db, "suite.edited", request=request, user_id=context.user.id, detail=f"{suite_id} {section_key} {op}")
    db.commit()
    anchor = f"?open={section_key}-{focus}#t-{section_key}-{focus}" if op != "delete" else ""
    return RedirectResponse(f"/suites/{suite_id}/customise{anchor}", status_code=303)


@router.post("/suites/{suite_id}/customise/draft", response_class=HTMLResponse,
             dependencies=[Depends(verify_csrf)])
async def draft_action(request: Request, suite_id: str, action: str = Form("preview"),
                       target_id: int | None = Form(None), note: str = Form(""),
                       db: Session = Depends(get_session),
                       context: AuthContext = Depends(require_user)):
    base, missing = _suite_or_404(request, suite_id, context)
    if missing:
        return missing
    view, user = current(), context.user

    if action == "discard":
        custom.discard_draft(db, user.id, suite_id)
        db.commit()
        return RedirectResponse(f"/suites/{suite_id}/customise", status_code=303)
    if action == "reset":
        custom.discard_draft(db, user.id, suite_id)
        count = custom.reset(db, user.id, suite_id)
        audit(db, "suite.reset", request=request, user_id=user.id, detail=suite_id)
        db.commit()
        return RedirectResponse(f"/suites/{suite_id}?reset={count}", status_code=303)

    text = custom.working_text(db, user.id, base)
    checked, problems = custom.check(base, text, view)

    def page(**extra):
        return _page(request, db, context, base, view=view, **extra)

    if checked is None:
        return page(error="Fix the problems marked below first.")

    if action == "save":
        existing = custom.active(db, user.id, suite_id)
        if existing and existing.fingerprint == custom.fingerprint(checked):
            custom.discard_draft(db, user.id, suite_id)
            db.commit()
            return page(notice="No change: this is already your saved version.")
        row = custom.save(db, user.id, base, text, checked, note)
        custom.discard_draft(db, user.id, suite_id)
        audit(db, "suite.customised", request=request, user_id=user.id, detail=f"{suite_id} {row.fingerprint}")
        db.commit()
        return RedirectResponse(f"/suites/{suite_id}?saved={row.fingerprint}", status_code=303)

    try:
        preview = _preview(checked, view)
    except RunError as exc:
        return page(error=str(exc))
    if action == "preview":
        return page(preview=preview)
    if action == "try":
        target = db.scalar(select(ModelTarget).where(
            ModelTarget.id == target_id, ModelTarget.user_id == user.id)) if target_id else None
        if target is None:
            return page(preview=preview, error="Choose one of your own models to try it on.")
        if checked.get("restricted") and target.provider in judge.CLOUD_PROVIDERS:
            return page(preview=preview, chosen_target=target.id,
                        error="This suite is restricted: it may only be tried on on-prem models.")
        try:
            if preview["uses_judge"] or checked.get("family") == "safety":
                judge.require_configured(restricted=bool(checked.get("restricted")))
            tried = await _try_on_model(checked, view, target, user.id)
        except RunError as exc:
            return page(preview=preview, chosen_target=target.id, error=str(exc))
        audit(db, "suite.tried", request=request, user_id=user.id, detail=f"{suite_id} on {target.label}")
        db.commit()
        return page(preview=preview, tried=tried, chosen_target=target.id)
    return page()


# ---- advanced: the whole suite as text -------------------------------------------------- #


@router.get("/suites/{suite_id}/customise/yaml", response_class=HTMLResponse)
def yaml_page(request: Request, suite_id: str, db: Session = Depends(get_session),
              context: AuthContext = Depends(require_user)):
    base, missing = _suite_or_404(request, suite_id, context)
    if missing:
        return missing
    return templates.TemplateResponse(request, "customise_yaml.html", {
        "user": context.user, "csrf_token": context.csrf_token, "suite": base, "active": "suite:" + base["id"],
        "text": custom.working_text(db, context.user.id, base), "problems": [], "error": None})


@router.post("/suites/{suite_id}/customise/yaml", response_class=HTMLResponse,
             dependencies=[Depends(verify_csrf)])
def yaml_apply(request: Request, suite_id: str, yaml_text: str = Form(""), action: str = Form("check"),
               db: Session = Depends(get_session), context: AuthContext = Depends(require_user)):
    """Edit the whole suite as text. Applying stores it as the draft (it can still be invalid);
    saving as a version always goes through the checked editor page."""
    base, missing = _suite_or_404(request, suite_id, context)
    if missing:
        return missing
    checked, problems = custom.check(base, yaml_text, current())
    error = None
    if action == "apply":
        try:
            custom.parse_yaml(yaml_text)
            custom.put_draft(db, context.user.id, suite_id, yaml_text)
            audit(db, "suite.edited", request=request, user_id=context.user.id, detail=f"{suite_id} yaml")
            db.commit()
            return RedirectResponse(f"/suites/{suite_id}/customise", status_code=303)
        except custom.CustomError as exc:
            error = str(exc)
    return templates.TemplateResponse(request, "customise_yaml.html", {
        "user": context.user, "csrf_token": context.csrf_token, "suite": base, "active": "suite:" + base["id"],
        "text": yaml_text, "problems": problems, "error": error, "valid": checked is not None})
