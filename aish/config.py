"""Application configuration.

Security-relevant defaults live here and are strict by default: the app refuses to
start in production without real secrets rather than falling back to a known value.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Generated per-process when unset. Fine for local development (sessions drop on
# restart); production is required to supply real values, enforced below.
_EPHEMERAL = secrets.token_urlsafe(48)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AISH_", env_file=".env", extra="ignore")

    env: str = "development"
    debug: bool = False

    # Signs session cookies and CSRF tokens.
    secret_key: str = _EPHEMERAL
    # Wraps stored provider API keys (AES-256-GCM). Separate from secret_key so a
    # cookie-signing key rotation does not destroy stored credentials.
    encryption_key: str = _EPHEMERAL

    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'aish.db'}"

    site_name: str = "EUIBA AI Safety Harness"
    public_host: str = "aish.devalier.com"

    # Cookies
    session_cookie: str = "aish_session"
    session_max_age_seconds: int = 8 * 60 * 60
    session_idle_timeout_seconds: int = 60 * 60
    cookie_secure: bool = True
    cookie_samesite: str = "strict"

    # Password policy (NIST SP 800-63B: length over composition rules).
    password_min_length: int = 12
    password_max_length: int = 1024

    # Argon2id work factors.
    argon2_time_cost: int = 3
    argon2_memory_cost_kib: int = 65536
    argon2_parallelism: int = 2

    # Brute-force controls.
    login_max_attempts: int = 5
    login_lockout_seconds: int = 15 * 60
    register_max_per_hour_per_ip: int = 5

    # Registration gate. When set, only these email domains may register.
    allowed_email_domains: str = ""
    require_admin_approval: bool = False

    # Outbound model calls.
    allow_private_endpoints: bool = True  # on-prem models live on private networks
    request_timeout_seconds: float = 120.0
    max_concurrent_requests: int = 4
    # Must exceed the largest run the instance is expected to accept. Sampled
    # EU-MMLU alone is 3,200 items (16 languages x 8 subjects x 25); a full run is
    # ~17,200. A run that would exceed this is refused, never silently shortened.
    max_items_per_run: int = 5000

    @field_validator("cookie_samesite")
    @classmethod
    def _samesite(cls, value: str) -> str:
        allowed = {"strict", "lax", "none"}
        if value.lower() not in allowed:
            raise ValueError(f"cookie_samesite must be one of {sorted(allowed)}")
        return value.lower()

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"production", "prod"}

    @property
    def email_domain_allowlist(self) -> set[str]:
        return {d.strip().lower().lstrip("@") for d in self.allowed_email_domains.split(",") if d.strip()}

    def enforce_production_invariants(self) -> None:
        """Fail closed rather than run production on development defaults."""
        if not self.is_production:
            return
        problems = []
        if self.secret_key == _EPHEMERAL or len(self.secret_key) < 32:
            problems.append("AISH_SECRET_KEY must be set to at least 32 characters")
        if self.encryption_key == _EPHEMERAL or len(self.encryption_key) < 32:
            problems.append("AISH_ENCRYPTION_KEY must be set to at least 32 characters")
        if self.secret_key == self.encryption_key:
            problems.append("AISH_SECRET_KEY and AISH_ENCRYPTION_KEY must differ")
        if not self.cookie_secure:
            problems.append("AISH_COOKIE_SECURE must stay true in production")
        if self.debug:
            problems.append("AISH_DEBUG must be false in production")
        if problems:
            raise RuntimeError("refusing to start in production: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.enforce_production_invariants()
    return settings


def running_under_tls(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return settings.cookie_secure and settings.is_production or os.environ.get("AISH_FORCE_TLS") == "1"
