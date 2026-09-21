"""Registration, login, logout and account management."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import User, UserSession, utcnow
from ..packs import euiba_groups, euiba_lookup
from ..security import (
    PasswordPolicyError,
    new_token,
    check_password_policy,
    dummy_verify,
    hash_password,
    is_valid_email,
    needs_rehash,
    normalise_email,
    verify_password,
)
from .deps import (
    CSRF_COOKIE,
    AuthContext,
    audit,
    client_ip,
    create_session,
    clear_session_cookies,
    ip_hash,
    optional_user,
    rate_limit,
    require_user,
    revoke_session,
    set_csrf_cookie,
    templates,
    verify_csrf,
)

router = APIRouter()

# One message for every failed login, so the form cannot be used to enumerate
# which email addresses are registered.
_LOGIN_FAILED = "Email or password is incorrect."


def _safe_next(raw: str | None) -> str:
    """Only same-site relative paths are accepted as a post-login redirect."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/dashboard"
    return raw


@router.get("/login", response_class=HTMLResponse)
def login_form(
    request: Request,
    next: str = "/dashboard",
    context: AuthContext | None = Depends(optional_user),
):
    if context:
        return RedirectResponse("/dashboard", status_code=303)
    token = request.cookies.get(CSRF_COOKIE) or new_token()
    page = templates.TemplateResponse(
        request, "login.html", {"next": _safe_next(next), "error": None, "csrf_token": token}
    )
    set_csrf_cookie(page, token)
    return page


@router.post("/login", dependencies=[Depends(verify_csrf)])
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/dashboard"),
    db: Session = Depends(get_session),
):
    settings = get_settings()
    destination = _safe_next(next)
    email = normalise_email(email)

    def failure(message: str = _LOGIN_FAILED, status_code: int = 401) -> Response:
        response = templates.TemplateResponse(
            request,
            "login.html",
            {"next": destination, "error": message,
             "csrf_token": request.cookies.get(CSRF_COOKIE, "")},
            status_code=status_code,
        )
        return response

    if not rate_limit(
        db, f"login:ip:{ip_hash(request)}", limit=20, window_seconds=settings.login_lockout_seconds
    ):
        audit(db, "login.rate_limited", request=request, detail="ip budget exhausted")
        return failure("Too many attempts from this address. Try again later.", 429)

    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        dummy_verify(settings)  # equalise response time for unknown accounts
        audit(db, "login.failed", request=request, detail="unknown account")
        return failure()

    now = utcnow()
    locked_until = user.locked_until
    if locked_until is not None:
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=now.tzinfo)
        if locked_until > now:
            audit(db, "login.locked", request=request, user_id=user.id)
            return failure("This account is temporarily locked. Try again later.", 429)

    if not verify_password(user.password_hash, password, settings):
        user.failed_logins += 1
        if user.failed_logins >= settings.login_max_attempts:
            user.locked_until = now + timedelta(seconds=settings.login_lockout_seconds)
            user.failed_logins = 0
            audit(db, "login.lockout", request=request, user_id=user.id)
        else:
            audit(db, "login.failed", request=request, user_id=user.id)
        return failure()

    if not user.is_active:
        audit(db, "login.inactive", request=request, user_id=user.id)
        return failure("This account is not active. Contact your administrator.", 403)
    if settings.require_admin_approval and user.approved_at is None:
        audit(db, "login.unapproved", request=request, user_id=user.id)
        return failure("This account is awaiting administrator approval.", 403)

    if needs_rehash(user.password_hash, settings):
        user.password_hash = hash_password(password, settings)

    user.failed_logins = 0
    user.locked_until = None
    user.last_login_at = now

    response = RedirectResponse(destination, status_code=303)
    create_session(db, user, request, response)
    audit(db, "login.success", request=request, user_id=user.id)
    return response


@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request, context: AuthContext | None = Depends(optional_user)):
    if context:
        return RedirectResponse("/dashboard", status_code=303)
    settings = get_settings()
    token = request.cookies.get(CSRF_COOKIE) or new_token()
    page = templates.TemplateResponse(
        request,
        "register.html",
        {
            "groups": euiba_groups(),
            "error": None,
            "form": {},
            "csrf_token": token,
            "min_length": settings.password_min_length,
            "domain_allowlist": sorted(settings.email_domain_allowlist),
        },
    )
    set_csrf_cookie(page, token)
    return page


@router.post("/register", dependencies=[Depends(verify_csrf)])
def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    euiba_id: str = Form(...),
    display_name: str = Form(""),
    db: Session = Depends(get_session),
):
    settings = get_settings()
    email = normalise_email(email)
    display_name = display_name.strip()[:120]
    euibas = euiba_lookup()

    def failure(message: str, status_code: int = 400) -> Response:
        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "groups": euiba_groups(),
                "error": message,
                "form": {"email": email, "euiba_id": euiba_id, "display_name": display_name},
                "csrf_token": request.cookies.get(CSRF_COOKIE, ""),
                "min_length": settings.password_min_length,
                "domain_allowlist": sorted(settings.email_domain_allowlist),
            },
            status_code=status_code,
        )

    if not rate_limit(
        db,
        f"register:ip:{ip_hash(request)}",
        limit=settings.register_max_per_hour_per_ip,
        window_seconds=3600,
    ):
        audit(db, "register.rate_limited", request=request)
        return failure("Too many registrations from this address. Try again later.", 429)

    if not is_valid_email(email):
        return failure("Enter a valid email address.")

    allowlist = settings.email_domain_allowlist
    if allowlist and email.rsplit("@", 1)[-1] not in allowlist:
        return failure("Registration is limited to approved institutional email domains.")

    if euiba_id not in euibas:
        return failure("Select your institution, body or agency from the list.")

    if password != password_confirm:
        return failure("The two passwords do not match.")

    try:
        check_password_policy(password, email, settings)
    except PasswordPolicyError as exc:
        return failure(str(exc))

    if db.scalar(select(User).where(User.email == email)) is not None:
        # Do not confirm that the address is taken; that would leak the user list.
        audit(db, "register.duplicate", request=request, detail="existing address")
        return templates.TemplateResponse(
            request, "register_submitted.html", {"email": email}, status_code=202
        )

    first_user = db.scalar(select(User).limit(1)) is None
    user = User(
        email=email,
        password_hash=hash_password(password, settings),
        display_name=display_name,
        euiba_id=euiba_id,
        euiba_name=euibas[euiba_id],
        is_admin=first_user,  # the account that bootstraps the instance administers it
        is_active=True,
        approved_at=utcnow() if (first_user or not settings.require_admin_approval) else None,
    )
    db.add(user)
    db.flush()
    audit(db, "register.success", request=request, user_id=user.id, detail=euiba_id)

    if settings.require_admin_approval and not first_user:
        return templates.TemplateResponse(request, "register_submitted.html", {"email": email})

    response = RedirectResponse("/dashboard", status_code=303)
    create_session(db, user, request, response)
    return response


@router.post("/logout", dependencies=[Depends(verify_csrf)])
def logout(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    revoke_session(db, context.session_record)
    audit(db, "logout", request=request, user_id=context.user.id)
    response = RedirectResponse("/login", status_code=303)
    clear_session_cookies(response)
    return response


@router.get("/account", response_class=HTMLResponse)
def account(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    sessions = list(
        db.scalars(
            select(UserSession)
            .where(UserSession.user_id == context.user.id, UserSession.revoked_at.is_(None))
            .order_by(UserSession.last_seen_at.desc())
        )
    )
    return templates.TemplateResponse(
        request,
        "account.html",
        {
            "user": context.user,
            "csrf_token": context.csrf_token,
            "sessions": sessions,
            "this_session_id": context.session_record.id,
            "error": None,
            "notice": None,
        },
    )


@router.post("/account/password", dependencies=[Depends(verify_csrf)])
def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    new_password_confirm: str = Form(...),
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    settings = get_settings()
    user = context.user

    def render(error: str | None = None, notice: str | None = None, status_code: int = 200):
        return templates.TemplateResponse(
            request,
            "account.html",
            {
                "user": user,
                "csrf_token": context.csrf_token,
                "sessions": [],
                "this_session_id": context.session_record.id,
                "error": error,
                "notice": notice,
            },
            status_code=status_code,
        )

    if not verify_password(user.password_hash, current_password, settings):
        audit(db, "password.change_failed", request=request, user_id=user.id)
        return render(error="Your current password is incorrect.", status_code=403)
    if new_password != new_password_confirm:
        return render(error="The two new passwords do not match.", status_code=400)
    try:
        check_password_policy(new_password, user.email, settings)
    except PasswordPolicyError as exc:
        return render(error=str(exc), status_code=400)

    user.password_hash = hash_password(new_password, settings)
    user.password_changed_at = utcnow()

    # A password change invalidates every other session (session fixation defence).
    others = db.scalars(
        select(UserSession).where(
            UserSession.user_id == user.id,
            UserSession.id != context.session_record.id,
            UserSession.revoked_at.is_(None),
        )
    )
    for record in others:
        record.revoked_at = utcnow()

    audit(db, "password.changed", request=request, user_id=user.id)
    return render(notice="Password changed. All other sessions have been signed out.")


@router.post("/account/sessions/revoke", dependencies=[Depends(verify_csrf)])
def revoke_other_sessions(
    request: Request,
    db: Session = Depends(get_session),
    context: AuthContext = Depends(require_user),
):
    others = db.scalars(
        select(UserSession).where(
            UserSession.user_id == context.user.id,
            UserSession.id != context.session_record.id,
            UserSession.revoked_at.is_(None),
        )
    )
    count = 0
    for record in others:
        record.revoked_at = utcnow()
        count += 1
    audit(db, "sessions.revoked", request=request, user_id=context.user.id, detail=str(count))
    return RedirectResponse("/account", status_code=303)
