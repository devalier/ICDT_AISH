"""Security middleware: response headers, host validation and request size limits."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from ..config import get_settings

# No inline scripts and no inline styles anywhere in this application: every chart
# is inline SVG driven by attributes, so the policy needs no 'unsafe-inline' escape.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "object-src 'none'",
        "require-trusted-types-for 'script'",
    ]
)

_MAX_BODY_BYTES = 1_000_000

# The container and orchestrator probe this by address, not by the public name, so
# host validation would fail every health check. The endpoint returns a fixed string
# and reveals nothing, so exempting it costs nothing.
_HOST_CHECK_EXEMPT = frozenset({"/healthz"})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        settings = get_settings()

        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
        # Authenticated pages carry evidence and run results; never let a shared
        # cache or the back button hold them.
        response.headers.setdefault("Cache-Control", "no-store")

        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        return response


class RequestGuardMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies and unexpected Host headers before routing."""

    def __init__(self, app, allowed_hosts: set[str]):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts

    async def dispatch(self, request: Request, call_next):
        if self.allowed_hosts and request.url.path not in _HOST_CHECK_EXEMPT:
            host = (request.headers.get("host") or "").split(":")[0].lower()
            if host and host not in self.allowed_hosts:
                return PlainTextResponse("Unrecognised host.", status_code=400)

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > _MAX_BODY_BYTES:
            return PlainTextResponse("Request body too large.", status_code=413)

        return await call_next(request)
