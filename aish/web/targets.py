"""The per-user model test pane: point the harness at your own models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import ModelTarget, utcnow
from ..packs import current
from ..providers import PROVIDERS, ProviderError, get_adapter
from ..providers.base import GenerationParams
from ..security import decrypt_secret, encrypt_secret, redact, secret_hint, validate_endpoint
from .deps import AuthContext, audit, require_user, templates, verify_csrf

router = APIRouter()


def _owned(db: Session, context: AuthContext, target_id: int) -> ModelTarget | None:
    """Every lookup is scoped to the owner, so an id from another account 404s."""
    return db.scalar(
        select(ModelTarget).where(
            ModelTarget.id == target_id, ModelTarget.user_id == context.user.id
        )
    )


def _render(
    request: Request,
    db: Session,
    context: AuthContext,
    *,
    error: str | None = None,
    notice: str | None = None,
    form: dict | None = None,
    status_code: int = 200,
):
    targets = list(
        db.scalars(
            select(ModelTarget)
            .where(ModelTarget.user_id == context.user.id)
            .order_by(ModelTarget.created_at.desc())
        )
    )
    return templates.TemplateResponse(
        request,
        "targets.html",
        {
            "user": context.user,
            "csrf_token": context.csrf_token,
            "targets": targets,
            "providers": PROVIDERS,
            "fleet": current().fleet(),
            "error": error,
            "notice": notice,
            "form": form or {},
        },
        status_code=status_code,
    )


@router.get("/models", response_class=HTMLResponse)
def list_targets(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    return _render(request, db, context)


@router.post("/models", dependencies=[Depends(verify_csrf)])
def create_target(
    request: Request,
    label: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    endpoint: str = Form(""),
    api_key: str = Form(""),
    fleet_model_id: str = Form(""),
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    label = label.strip()[:120]
    model_name = model_name.strip()[:160]
    endpoint = endpoint.strip()[:400]
    api_key = api_key.strip()
    form = {
        "label": label, "provider": provider, "model_name": model_name,
        "endpoint": endpoint, "fleet_model_id": fleet_model_id,
    }

    def fail(message: str):
        return _render(request, db, context, error=message, form=form, status_code=400)

    if provider not in PROVIDERS:
        return fail("Choose a provider from the list.")
    spec = PROVIDERS[provider]
    if not label:
        return fail("Give this target a name.")
    if not model_name:
        return fail("Enter the model identifier the provider expects.")
    if spec["requires_key"] and not api_key:
        return fail("This provider requires an API key.")
    if spec["requires_endpoint"] and not endpoint:
        return fail("An on-prem target needs its endpoint URL.")
    endpoint_warning = ""
    if endpoint:
        check = validate_endpoint(endpoint)
        if not check.ok and not check.unresolved:
            return fail(f"Endpoint rejected: {check.reason}")
        if check.unresolved:
            # An internal DNS name may not resolve from the web tier. Saving it is
            # allowed; the same check runs again before every outbound request, and
            # that is the one that decides.
            endpoint_warning = (
                " The host does not resolve from this server yet, so test the "
                "connection before relying on it."
            )
    if db.scalar(
        select(ModelTarget).where(
            ModelTarget.user_id == context.user.id, ModelTarget.label == label
        )
    ):
        return fail("You already have a target with that name.")

    target = ModelTarget(
        user_id=context.user.id,
        label=label,
        provider=provider,
        model_name=model_name,
        endpoint=endpoint,
        fleet_model_id=fleet_model_id[:64],
    )
    if api_key:
        # Encrypted immediately and bound to this owner. The plaintext is never
        # written to the database, a log, or a template.
        target.api_key_ciphertext = encrypt_secret(api_key, f"user:{context.user.id}")
        target.api_key_hint = secret_hint(api_key)

    db.add(target)
    db.flush()
    audit(db, "target.created", request=request, user_id=context.user.id,
          detail=f"{provider}:{model_name}")
    if endpoint_warning:
        return _render(request, db, context, notice=f"Added {label}.{endpoint_warning}")
    return RedirectResponse("/models", status_code=303)


@router.post("/models/{target_id}/key", dependencies=[Depends(verify_csrf)])
def rotate_key(
    request: Request,
    target_id: int,
    api_key: str = Form(""),
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    target = _owned(db, context, target_id)
    if target is None:
        return _render(request, db, context, error="That target does not exist.", status_code=404)

    api_key = api_key.strip()
    if api_key:
        target.api_key_ciphertext = encrypt_secret(api_key, f"user:{context.user.id}")
        target.api_key_hint = secret_hint(api_key)
        detail = "key replaced"
    else:
        target.api_key_ciphertext = None
        target.api_key_hint = ""
        detail = "key removed"
    target.updated_at = utcnow()
    audit(db, "target.key_rotated", request=request, user_id=context.user.id, detail=detail)
    return _render(request, db, context, notice=f"Credential updated for {target.label}.")


@router.post("/models/{target_id}/delete", dependencies=[Depends(verify_csrf)])
def delete_target(
    request: Request,
    target_id: int,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    target = _owned(db, context, target_id)
    if target is None:
        return _render(request, db, context, error="That target does not exist.", status_code=404)
    label = target.label
    db.delete(target)
    audit(db, "target.deleted", request=request, user_id=context.user.id, detail=label)
    return _render(request, db, context, notice=f"Removed {label} and its runs.")


@router.post("/models/{target_id}/check", dependencies=[Depends(verify_csrf)])
async def check_target(
    request: Request,
    target_id: int,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    """One minimal live call, so a user finds out here rather than mid-run."""
    target = _owned(db, context, target_id)
    if target is None:
        return _render(request, db, context, error="That target does not exist.", status_code=404)

    settings = get_settings()
    api_key = (
        decrypt_secret(target.api_key_ciphertext, f"user:{context.user.id}")
        if target.api_key_ciphertext
        else None
    )
    ok, message = False, ""
    try:
        completion = await get_adapter(target.provider).generate(
            model=target.model_name,
            prompt="Reply with the single word: ready",
            system=None,
            api_key=api_key,
            endpoint=target.endpoint or None,
            params=GenerationParams(max_tokens=16),
            timeout=min(settings.request_timeout_seconds, 30.0),
        )
        ok = True
        message = f"Responded in {completion.latency_ms} ms."
    except ProviderError as exc:
        message = redact(str(exc), api_key or "")
    except Exception as exc:  # noqa: BLE001
        message = redact(f"{type(exc).__name__}: {exc}", api_key or "")

    target.last_checked_at = utcnow()
    target.last_check_ok = ok
    target.last_check_message = message[:400]
    audit(db, "target.checked", request=request, user_id=context.user.id,
          detail=f"{target.label}: {'ok' if ok else 'failed'}")

    return _render(
        request, db, context,
        notice=f"{target.label}: {message}" if ok else None,
        error=None if ok else f"{target.label}: {message}",
    )
