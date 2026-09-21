"""End-to-end tests for the web application, weighted toward the security controls."""

from __future__ import annotations

import pytest

from tests.conftest import csrf_of


# --------------------------------------------------------------------------- #
# Registration and login
# --------------------------------------------------------------------------- #


def test_register_then_land_on_dashboard(client, register):
    response = register(client)
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Model comparison" in dashboard.text


def test_registration_requires_a_listed_euiba(client):
    client.get("/register")
    response = client.post(
        "/register",
        data={
            "csrf_token": csrf_of(client),
            "email": "user@ec.europa.eu",
            "password": "a-long-enough-passphrase",
            "password_confirm": "a-long-enough-passphrase",
            "euiba_id": "NOT_A_REAL_BODY",
        },
    )
    assert response.status_code == 400
    assert "Select your institution" in response.text


def test_registration_enforces_password_length(client):
    client.get("/register")
    response = client.post(
        "/register",
        data={
            "csrf_token": csrf_of(client),
            "email": "user@ec.europa.eu",
            "password": "short",
            "password_confirm": "short",
            "euiba_id": "COM",
        },
    )
    assert response.status_code == 400
    assert "at least 12 characters" in response.text


def test_duplicate_registration_does_not_disclose_the_account(client, register):
    register(client)
    client.cookies.clear()
    client.get("/register")
    response = client.post(
        "/register",
        data={
            "csrf_token": csrf_of(client),
            "email": "user@ec.europa.eu",
            "password": "another-long-passphrase",
            "password_confirm": "another-long-passphrase",
            "euiba_id": "COM",
        },
    )
    # Same shape as a successful pending registration: no "already taken" signal.
    assert response.status_code == 202
    assert "already" not in response.text.lower()


def test_login_failure_message_is_identical_for_unknown_and_wrong_password(client, register):
    register(client)
    client.cookies.clear()

    client.get("/login")
    unknown = client.post(
        "/login",
        data={"csrf_token": csrf_of(client), "email": "nobody@ec.europa.eu",
              "password": "a-long-enough-passphrase", "next": "/dashboard"},
    )
    client.get("/login")
    wrong = client.post(
        "/login",
        data={"csrf_token": csrf_of(client), "email": "user@ec.europa.eu",
              "password": "wrong-but-long-enough", "next": "/dashboard"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert "Email or password is incorrect." in unknown.text
    assert "Email or password is incorrect." in wrong.text


def test_account_locks_after_repeated_failures(client, register):
    register(client)
    client.cookies.clear()
    for _ in range(5):
        client.get("/login")
        client.post(
            "/login",
            data={"csrf_token": csrf_of(client), "email": "user@ec.europa.eu",
                  "password": "wrong-but-long-enough", "next": "/dashboard"},
        )
    client.get("/login")
    response = client.post(
        "/login",
        data={"csrf_token": csrf_of(client), "email": "user@ec.europa.eu",
              "password": "a-long-enough-passphrase", "next": "/dashboard"},
    )
    assert response.status_code == 429
    assert "locked" in response.text.lower()


def test_logout_revokes_the_session(client, register):
    register(client)
    token = client.cookies.get("aish_session")
    client.post("/logout", data={"csrf_token": csrf_of(client)}, follow_redirects=False)

    client.cookies.set("aish_session", token)
    response = client.get("/dashboard", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


def test_password_change_signs_out_other_sessions(client, register):
    from fastapi.testclient import TestClient

    from aish.app import create_app

    register(client)
    app = client.app
    with TestClient(app, base_url="http://testserver") as second:
        second.get("/login")
        second.post(
            "/login",
            data={"csrf_token": csrf_of(second), "email": "user@ec.europa.eu",
                  "password": "a-long-enough-passphrase", "next": "/dashboard"},
            follow_redirects=False,
        )
        assert second.get("/dashboard").status_code == 200

        client.post(
            "/account/password",
            data={
                "csrf_token": csrf_of(client),
                "current_password": "a-long-enough-passphrase",
                "new_password": "a-different-long-passphrase",
                "new_password_confirm": "a-different-long-passphrase",
            },
        )
        stale = second.get("/dashboard", follow_redirects=False)
        assert stale.status_code == 303


# --------------------------------------------------------------------------- #
# CSRF, headers, access control
# --------------------------------------------------------------------------- #


def test_post_without_csrf_token_is_rejected(client, register):
    register(client)
    response = client.post("/models", data={"label": "x", "provider": "openai",
                                            "model_name": "gpt-5.1", "api_key": "k"})
    assert response.status_code == 403


def test_post_with_a_mismatched_csrf_token_is_rejected(client, register):
    register(client)
    response = client.post(
        "/models",
        data={"csrf_token": "not-the-right-token", "label": "x", "provider": "openai",
              "model_name": "gpt-5.1", "api_key": "k"},
    )
    assert response.status_code == 403


def test_security_headers_are_present(client):
    response = client.get("/login")
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["cache-control"] == "no-store"


def test_session_cookie_is_httponly_and_samesite(client, register):
    response = register(client)
    cookies = response.headers.get_list("set-cookie")
    session_cookie = next(c for c in cookies if c.startswith("aish_session="))
    assert "HttpOnly" in session_cookie
    assert "samesite=strict" in session_cookie.lower()


def test_unauthenticated_pages_redirect_to_login(client):
    for path in ["/dashboard", "/models", "/runs", "/panes/bias", "/panes/scoring",
                 "/panes/parity", "/panes/suites", "/account"]:
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303, path
        assert response.headers["location"].startswith("/login"), path


def test_unknown_host_is_refused(client):
    response = client.get("/login", headers={"host": "evil.example.com"})
    assert response.status_code == 400


def test_open_redirect_is_not_possible_after_login(client, register):
    register(client)
    client.post("/logout", data={"csrf_token": csrf_of(client)})
    client.get("/login")
    response = client.post(
        "/login",
        data={"csrf_token": csrf_of(client), "email": "user@ec.europa.eu",
              "password": "a-long-enough-passphrase", "next": "//evil.example.com/"},
        follow_redirects=False,
    )
    assert response.headers["location"] == "/dashboard"


# --------------------------------------------------------------------------- #
# Model targets and credential handling
# --------------------------------------------------------------------------- #


def _add_target(client, **overrides):
    payload = {
        "csrf_token": csrf_of(client),
        "label": "My Llama",
        "provider": "openai_compatible",
        "model_name": "llama-3.3-70b-instruct",
        "endpoint": "https://example.com/v1",
        "api_key": "sk-secret-value-9999",
        "fleet_model_id": "llama-3.3-70b",
    }
    payload.update(overrides)
    return client.post("/models", data=payload, follow_redirects=False)


def test_api_key_is_encrypted_and_never_rendered(client, register):
    from sqlalchemy import select

    from aish.db import session_scope
    from aish.models import ModelTarget

    register(client)
    assert _add_target(client).status_code == 303

    page = client.get("/models")
    assert "sk-secret-value-9999" not in page.text
    assert "****9999" in page.text

    with session_scope() as db:
        target = db.scalar(select(ModelTarget))
        assert target.api_key_ciphertext
        assert "sk-secret-value-9999" not in target.api_key_ciphertext


def test_stored_key_decrypts_only_for_its_owner(client, register):
    from sqlalchemy import select

    from aish.db import session_scope
    from aish.models import ModelTarget
    from aish.security import DecryptionError, decrypt_secret

    register(client)
    _add_target(client)
    with session_scope() as db:
        target = db.scalar(select(ModelTarget))
        assert decrypt_secret(target.api_key_ciphertext, f"user:{target.user_id}") == "sk-secret-value-9999"
        with pytest.raises(DecryptionError):
            decrypt_secret(target.api_key_ciphertext, "user:999")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8000/v1",
        "http://localhost/v1",
        "http://169.254.169.254/latest/meta-data",
        "file:///etc/passwd",
        "https://user:pass@example.com/v1",
    ],
)
def test_ssrf_prone_endpoints_are_refused(client, register, endpoint):
    register(client)
    response = _add_target(client, endpoint=endpoint, label=f"t-{abs(hash(endpoint))}")
    assert response.status_code == 400
    assert "rejected" in response.text.lower() or "not a valid" in response.text.lower()


def test_one_user_cannot_see_another_users_target(client, register):
    from fastapi.testclient import TestClient

    register(client, email="first@ec.europa.eu")
    _add_target(client, label="First target")

    with TestClient(client.app, base_url="http://testserver") as other:
        other.get("/register")
        other.post(
            "/register",
            data={
                "csrf_token": csrf_of(other),
                "email": "second@ec.europa.eu",
                "password": "a-long-enough-passphrase",
                "password_confirm": "a-long-enough-passphrase",
                "euiba_id": "EP",
            },
            follow_redirects=False,
        )
        page = other.get("/models")
        assert "First target" not in page.text

        # Acting on someone else's id must not succeed.
        deleted = other.post("/models/1/delete", data={"csrf_token": csrf_of(other)})
        assert deleted.status_code == 404


def test_one_user_cannot_read_another_users_run(client, register):
    from fastapi.testclient import TestClient

    from aish.db import session_scope
    from aish.models import Run

    register(client, email="first@ec.europa.eu")
    _add_target(client)
    with session_scope() as db:
        db.add(Run(user_id=1, target_id=1, suite_id="euiba_bias", suite_version="0.1.0",
                   family="bias", status="done", items_total=1, items_done=1))

    with TestClient(client.app, base_url="http://testserver") as other:
        other.get("/register")
        other.post(
            "/register",
            data={"csrf_token": csrf_of(other), "email": "second@ec.europa.eu",
                  "password": "a-long-enough-passphrase",
                  "password_confirm": "a-long-enough-passphrase", "euiba_id": "EP"},
            follow_redirects=False,
        )
        assert other.get("/runs/1").status_code == 404
        assert other.get("/runs/1/evidence").status_code == 404
        assert other.get("/runs/1/status").status_code == 404


# --------------------------------------------------------------------------- #
# Panes
# --------------------------------------------------------------------------- #


def test_bias_pane_states_its_legal_basis_before_any_number(client, register):
    register(client)
    page = client.get("/panes/bias")
    assert page.status_code == 200
    assert "Charter Art. 21" in page.text
    assert "TFEU Art. 18" in page.text
    assert "Nationality (EU Member State)" in page.text
    assert "Known limits of this method" in page.text


def test_scoring_pane_shows_thresholds_and_lexicons(client, register):
    register(client)
    page = client.get("/panes/scoring")
    assert page.status_code == 200
    assert "rate_disparity" in page.text
    assert "parity_floor" in page.text
    assert "i don&#39;t know" in page.text or "i don't know" in page.text


def test_suite_pane_renders_the_live_definition(client, register):
    register(client)
    page = client.get("/panes/suites/euiba_bias")
    assert page.status_code == 200
    assert "counterfactual_disparity" in page.text
    assert "changelog" in page.text.lower()


def test_parity_pane_loads_without_runs(client, register):
    register(client)
    page = client.get("/panes/parity")
    assert page.status_code == 200
    assert "parity floor" in page.text.lower()


def test_the_health_probe_answers_without_the_public_host(client):
    """The container health check probes by address. Host validation must not fail
    it, or the container is permanently unhealthy in production."""
    response = client.get("/healthz", headers={"host": "127.0.0.1:8000"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_under_production_hosts_only_healthz_is_exempt(monkeypatch):
    """Development allows localhost, so the exemption has to be checked against a
    production configuration — which is where it matters."""
    from fastapi.testclient import TestClient

    from aish import config
    from aish.app import create_app

    monkeypatch.setenv("AISH_ENV", "production")
    monkeypatch.setenv("AISH_SECRET_KEY", "p" * 40)
    monkeypatch.setenv("AISH_ENCRYPTION_KEY", "q" * 40)
    monkeypatch.setenv("AISH_PUBLIC_HOST", "aish.devalier.com")
    monkeypatch.setenv("AISH_COOKIE_SECURE", "true")  # the fixture relaxes this for dev
    config.get_settings.cache_clear()
    try:
        with TestClient(create_app(), base_url="http://127.0.0.1:8000") as probe:
            assert probe.get("/healthz").status_code == 200
            for path in ["/", "/login", "/static/app.css"]:
                assert probe.get(path, follow_redirects=False).status_code == 400, path
            # And the real public host is served normally.
            assert probe.get(
                "/login", headers={"host": "aish.devalier.com"}
            ).status_code == 200
    finally:
        config.get_settings.cache_clear()
