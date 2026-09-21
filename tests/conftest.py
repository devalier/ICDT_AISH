"""Test fixtures: an isolated database and a fresh settings cache per test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AISH_ENV", "development")
    monkeypatch.setenv("AISH_SECRET_KEY", "test-secret-key-that-is-long-enough-000000")
    monkeypatch.setenv("AISH_ENCRYPTION_KEY", "test-encryption-key-that-is-long-enough-1")
    monkeypatch.setenv("AISH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("AISH_COOKIE_SECURE", "false")
    # Argon2 at production cost makes a login test take seconds; the algorithm and
    # code path are identical, only the work factor is lowered.
    monkeypatch.setenv("AISH_ARGON2_TIME_COST", "1")
    monkeypatch.setenv("AISH_ARGON2_MEMORY_COST_KIB", "8192")

    from aish import config, db, security

    config.get_settings.cache_clear()
    security._DUMMY_HASH = None
    db.reset_engine_for_tests()
    db.init_db()
    yield
    db.reset_engine_for_tests()
    config.get_settings.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from aish.app import create_app

    with TestClient(create_app(), base_url="http://testserver") as test_client:
        yield test_client


def csrf_of(client) -> str:
    return client.cookies.get("aish_csrf", "")


@pytest.fixture
def register():
    def _register(client, email="user@ec.europa.eu", password="a-long-enough-passphrase"):
        client.get("/register")
        return client.post(
            "/register",
            data={
                "csrf_token": csrf_of(client),
                "email": email,
                "password": password,
                "password_confirm": password,
                "euiba_id": "COM",
                "display_name": "Test User",
            },
            follow_redirects=False,
        )

    return _register
