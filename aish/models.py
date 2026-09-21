"""ORM models.

Stored provider API keys are ciphertext only (``ModelTarget.api_key_ciphertext``);
the plaintext exists in memory for the duration of a single outbound call and is
never written to the database, to a log, or into a template.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(120), default="")
    euiba_id: Mapped[str] = mapped_column(String(64))
    euiba_name: Mapped[str] = mapped_column(String(200))

    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    password_changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    sessions: Mapped[list["UserSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    targets: Mapped[list["ModelTarget"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(Base):
    """Server-side session record. The cookie carries an opaque token; this row
    holds only its SHA-256 digest, so database access alone cannot forge a login."""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    ip_hash: Mapped[str] = mapped_column(String(64), default="")
    user_agent: Mapped[str] = mapped_column(String(256), default="")

    user: Mapped[User] = relationship(back_populates="sessions")


class ModelTarget(Base):
    """A model a user has pointed the harness at, with their own credential."""

    __tablename__ = "model_targets"
    __table_args__ = (UniqueConstraint("user_id", "label", name="uq_target_label_per_user"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    label: Mapped[str] = mapped_column(String(120))
    fleet_model_id: Mapped[str] = mapped_column(String(64), default="")  # packs/models.yaml id
    provider: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(160))
    endpoint: Mapped[str] = mapped_column(String(400), default="")

    api_key_ciphertext: Mapped[str | None] = mapped_column(Text, default=None)
    api_key_hint: Mapped[str] = mapped_column(String(16), default="")

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_check_ok: Mapped[bool | None] = mapped_column(Boolean, default=None)
    last_check_message: Mapped[str] = mapped_column(String(400), default="")

    user: Mapped[User] = relationship(back_populates="targets")
    runs: Mapped[list["Run"]] = relationship(back_populates="target", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("model_targets.id", ondelete="CASCADE"), index=True
    )

    suite_id: Mapped[str] = mapped_column(String(64))
    suite_version: Mapped[str] = mapped_column(String(32))
    family: Mapped[str] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(24), default="queued")  # queued|running|done|failed|cancelled
    sample_mode: Mapped[bool] = mapped_column(Boolean, default=True)
    items_total: Mapped[int] = mapped_column(Integer, default=0)
    items_done: Mapped[int] = mapped_column(Integer, default=0)

    manifest_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(String(600), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    target: Mapped[ModelTarget] = relationship(back_populates="runs")
    results: Mapped[list["Result"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Result(Base):
    """One prompt/response pair and its score. This is the evidence trail —
    every number in the UI is one click from these rows."""

    __tablename__ = "results"
    __table_args__ = (Index("ix_results_run_stratum", "run_id", "stratum"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    item_id: Mapped[str] = mapped_column(String(160))
    stratum: Mapped[str] = mapped_column(String(120), default="")
    variant: Mapped[str] = mapped_column(String(120), default="")

    prompt: Mapped[str] = mapped_column(Text, default="")
    response: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str] = mapped_column(String(40), default="")
    score: Mapped[float | None] = mapped_column(default=None)

    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(String(400), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="results")


class AuditEvent(Base):
    """Security-relevant events. Retained for audit; contains no secrets."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), default=None, index=True
    )
    event: Mapped[str] = mapped_column(String(64), index=True)
    ip_hash: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[str] = mapped_column(String(600), default="")


class RateLimitBucket(Base):
    """Durable counter for login/registration throttling, so a restart does not
    reset an attacker's budget."""

    __tablename__ = "rate_limit_buckets"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
