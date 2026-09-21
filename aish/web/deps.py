"""Request-scoped helpers: authentication, CSRF, rate limiting and templating."""

from __future__ import annotations

import hashlib
from datetime import timedelta

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import BASE_DIR, Settings, get_settings
from ..db import get_session
from ..models import AuditEvent, RateLimitBucket, User, UserSession, utcnow
from ..security import constant_time_equals, new_token, token_fingerprint

CSRF_COOKIE = "aish_csrf"
CSRF_FIELD = "csrf_token"

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.autoescape = True  # explicit: every template variable is escaped


# --------------------------------------------------------------------------- #
# Client identity (hashed — the raw IP is never stored)
# --------------------------------------------------------------------------- #


def client_ip(request: Request) -> str:
    """Trust the proxy's forwarded address only for the first hop.

    The deployment terminates TLS at a reverse proxy we control; anything beyond
    the first entry in X-Forwarded-For is attacker-controlled and ignored.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else ""


def ip_hash(request: Request, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    raw = f"{settings.secret_key}|{client_ip(request)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def audit(
    db: Session, event: str, *, request: Request, user_id: int | None = None, detail: str = ""
) -> None:
    db.add(
        AuditEvent(
            event=event,
            user_id=user_id,
            ip_hash=ip_hash(request),
            detail=detail[:600],
        )
    )


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def rate_limit(db: Session, key: str, *, limit: int, window_seconds: int) -> bool:
    """Fixed-window counter persisted in the database.

    Durable on purpose: a restart must not hand an attacker a fresh budget.
    Returns True when the request is within budget.
    """
    now = utcnow()
    bucket = db.scalar(select(RateLimitBucket).where(RateLimitBucket.key == key[:180]))
    if bucket is None:
        db.add(RateLimitBucket(key=key[:180], count=1, window_start=now))
        return True

    window_start = bucket.window_start
    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=now.tzinfo)
    if (now - window_start).total_seconds() > window_seconds:
        bucket.count = 1
        bucket.window_start = now
        return True

    bucket.count += 1
    return bucket.count <= limit


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def create_session(db: Session, user: User, request: Request, response: Response) -> UserSession:
    settings = get_settings()
    token = new_token()
    csrf = new_token()
    record = UserSession(
        user_id=user.id,
        token_hash=token_fingerprint(token),
        csrf_token=csrf,
        expires_at=utcnow() + timedelta(seconds=settings.session_max_age_seconds),
        ip_hash=ip_hash(request, settings),
        user_agent=(request.headers.get("user-agent") or "")[:256],
    )
    db.add(record)
    db.flush()

    response.set_cookie(
        settings.session_cookie,
        token,
        max_age=settings.session_max_age_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )
    set_csrf_cookie(response, csrf)
    return record


def set_csrf_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        CSRF_COOKIE,
        token,
        max_age=settings.session_max_age_seconds,
        httponly=True,  # the token reaches the browser only inside a rendered form
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


def clear_session_cookies(response: Response) -> None:
    settings = get_settings()
    for name in (settings.session_cookie, CSRF_COOKIE):
        response.delete_cookie(name, path="/")


def revoke_session(db: Session, record: UserSession) -> None:
    record.revoked_at = utcnow()


def load_session(request: Request, db: Session) -> UserSession | None:
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie)
    if not token:
        return None

    record = db.scalar(
        select(UserSession).where(UserSession.token_hash == token_fingerprint(token))
    )
    if record is None or record.revoked_at is not None:
        return None

    now = utcnow()
    expires_at = record.expires_at
    last_seen = record.last_seen_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=now.tzinfo)
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=now.tzinfo)

    if expires_at <= now:
        return None
    if (now - last_seen).total_seconds() > settings.session_idle_timeout_seconds:
        record.revoked_at = now
        return None

    record.last_seen_at = now
    return record


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


class AuthContext:
    def __init__(self, user: User, session_record: UserSession):
        self.user = user
        self.session_record = session_record

    @property
    def csrf_token(self) -> str:
        return self.session_record.csrf_token


def optional_user(request: Request, db: Session = Depends(get_session)) -> AuthContext | None:
    record = load_session(request, db)
    if record is None:
        return None
    user = db.get(User, record.user_id)
    if user is None or not user.is_active:
        return None
    return AuthContext(user, record)


def require_user(
    request: Request, context: AuthContext | None = Depends(optional_user)
) -> AuthContext:
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login?next=" + request.url.path},
        )
    return context


def require_admin(context: AuthContext = Depends(require_user)) -> AuthContext:
    if not context.user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrators only.")
    return context


async def verify_csrf(request: Request) -> None:
    """Double-submit CSRF check on every state-changing request.

    Paired with a SameSite=Strict session cookie, so a cross-site POST has neither
    the cookie nor the token.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    cookie_token = request.cookies.get(CSRF_COOKIE, "")
    form = await request.form()
    form_token = str(form.get(CSRF_FIELD, ""))
    if not cookie_token or not form_token or not constant_time_equals(cookie_token, form_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security token missing or invalid. Reload the page and try again.",
        )




# --------------------------------------------------------------------------- #
# Template filters — presentation only
# --------------------------------------------------------------------------- #


def _fmt_pct(value, places: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(value, places: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_when(value) -> str:
    if value is None:
        return "—"
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _bar_width(value, maximum: float = 1.0, full: float = 100.0) -> float:
    """Bar length as a percentage of the track. Clamped so a bad value cannot
    draw outside the chart."""
    try:
        ratio = float(value) / float(maximum) if maximum else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return max(0.0, min(1.0, ratio)) * full


def _seq_step(value, low: float = 0.0, high: float = 1.0) -> int:
    """Quantise a continuous value onto the 5-step sequential ramp.

    Quantising is what lets a heatmap keep a strict CSP: each step is a class in
    the stylesheet rather than an inline fill.
    """
    if value is None:
        return 0
    try:
        ratio = (float(value) - low) / (high - low) if high > low else 0.0
    except (TypeError, ValueError):
        return 0
    ratio = max(0.0, min(1.0, ratio))
    return min(5, max(1, int(ratio * 5) + 1))


templates.env.filters["pct"] = _fmt_pct
templates.env.filters["num"] = _fmt_num
templates.env.filters["when"] = _fmt_when
templates.env.filters["bar"] = _bar_width
templates.env.filters["seq"] = _seq_step
