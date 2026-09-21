"""Application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR, get_settings
from .db import init_db
from .web import auth, panes, runs, targets
from .web.deps import templates
from .web.middleware import RequestGuardMiddleware, SecurityHeadersMiddleware

log = logging.getLogger("aish")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("%s started (env=%s)", app.title, get_settings().env)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        lifespan=lifespan,
        title=settings.site_name,
        docs_url=None,       # no interactive API explorer on an authenticated app
        redoc_url=None,
        openapi_url=None,
    )

    allowed_hosts = {settings.public_host.lower()}
    if not settings.is_production:
        allowed_hosts |= {"localhost", "127.0.0.1", "testserver"}

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestGuardMiddleware, allowed_hosts=allowed_hosts)

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    app.include_router(auth.router)
    app.include_router(targets.router)
    app.include_router(runs.router)
    app.include_router(panes.router)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
        # A redirect raised by require_user arrives here.
        if exc.status_code == 303 and "Location" in (exc.headers or {}):
            return RedirectResponse(exc.headers["Location"], status_code=303)
        if exc.status_code == 404:
            return templates.TemplateResponse(
                request, "not_found.html", {"user": None}, status_code=404
            )
        return templates.TemplateResponse(
            request,
            "error.html",
            {"user": None, "status": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> Response:
        # Log the detail; show the user nothing but a reference.
        log.exception("unhandled error on %s", request.url.path)
        return templates.TemplateResponse(
            request,
            "error.html",
            {"user": None, "status": 500, "detail": "An unexpected error occurred."},
            status_code=500,
        )

    return app


app = create_app()
