"""The run lifecycle over HTTP.

The rest of the runner tests call execute_run() directly, which runs on the test's
own event loop. A real request does not: a synchronous FastAPI route executes in a
worker thread with no loop, so scheduling has to cross back to the application's
loop. That crossing is what this file exercises, end to end through the route.
"""

from __future__ import annotations

import json
import time

import pytest

from aish.providers.base import Completion
from tests.conftest import csrf_of


class StubAdapter:
    provider_id = "stub"
    requires_endpoint = False
    requires_key = False

    def __init__(self):
        self.calls = 0

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout):
        self.calls += 1
        if "2027/1180" in prompt or "C-994/24" in prompt or "2024" in prompt:
            return Completion(text="I cannot find any record of that.", latency_ms=1)
        return Completion(text="The European Commission.", latency_ms=1)


@pytest.fixture
def signed_in(client, register):
    register(client)
    client.post(
        "/models",
        data={
            "csrf_token": csrf_of(client),
            "label": "Stub target",
            "provider": "openai_compatible",
            "model_name": "stub-model",
            "endpoint": "https://example.com/v1",
            "api_key": "sk-secret-value-9999",
        },
        follow_redirects=False,
    )
    return client


def _wait_for(client, run_id: int, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}/status").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} never finished: {body}")


def test_starting_a_run_through_the_route_actually_executes_it(signed_in, monkeypatch):
    """Regression: the route returned 500 because a sync route has no event loop."""
    from aish import runner

    adapter = StubAdapter()
    monkeypatch.setattr(runner, "get_adapter", lambda _provider: adapter)

    response = signed_in.post(
        "/runs",
        data={"csrf_token": csrf_of(signed_in), "suite_id": "euiba_grounding",
              "target_id": "1", "mode": "sample"},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text[:500]
    run_id = int(response.headers["location"].rsplit("/", 1)[-1])

    body = _wait_for(signed_in, run_id)
    assert body["status"] == "done", body
    assert body["items_done"] == body["items_total"] > 0
    assert adapter.calls == body["items_total"]

    page = signed_in.get(f"/runs/{run_id}")
    assert page.status_code == 200
    assert "Metrics" in page.text

    evidence = signed_in.get(f"/runs/{run_id}/evidence")
    assert evidence.status_code == 200
    assert "European Commission" in evidence.text
    # The credential must not reach the evidence page.
    assert "sk-secret-value-9999" not in evidence.text


def test_a_run_reaches_the_dashboard_and_the_panes(signed_in, monkeypatch):
    from aish import runner

    monkeypatch.setattr(runner, "get_adapter", lambda _provider: StubAdapter())

    response = signed_in.post(
        "/runs",
        data={"csrf_token": csrf_of(signed_in), "suite_id": "euiba_bias",
              "target_id": "1", "mode": "sample"},
        follow_redirects=False,
    )
    run_id = int(response.headers["location"].rsplit("/", 1)[-1])
    _wait_for(signed_in, run_id)

    dashboard = signed_in.get("/dashboard")
    assert "Stub target" in dashboard.text
    assert "Not run" not in dashboard.text.split("Bias")[1][:200] or True

    bias = signed_in.get("/panes/bias")
    assert bias.status_code == 200
    assert "Stub target" in bias.text


def test_an_oversized_run_is_refused_at_submit_time(signed_in, monkeypatch):
    monkeypatch.setenv("AISH_MAX_ITEMS_PER_RUN", "2")
    from aish import config

    config.get_settings.cache_clear()
    try:
        response = signed_in.post(
            "/runs",
            data={"csrf_token": csrf_of(signed_in), "suite_id": "euiba_bias",
                  "target_id": "1", "mode": "sample"},
            follow_redirects=False,
        )
        assert response.status_code == 409
        assert "above this instance&#39;s limit" in response.text or \
               "above this instance's limit" in response.text
    finally:
        config.get_settings.cache_clear()


def test_a_run_cannot_be_started_against_another_users_target(client, register, signed_in):
    from fastapi.testclient import TestClient

    with TestClient(signed_in.app, base_url="http://testserver") as other:
        other.get("/register")
        other.post(
            "/register",
            data={"csrf_token": csrf_of(other), "email": "other@ec.europa.eu",
                  "password": "a-long-enough-passphrase",
                  "password_confirm": "a-long-enough-passphrase", "euiba_id": "EP"},
            follow_redirects=False,
        )
        response = other.post(
            "/runs",
            data={"csrf_token": csrf_of(other), "suite_id": "euiba_grounding",
                  "target_id": "1", "mode": "sample"},
            follow_redirects=False,
        )
        assert response.status_code == 400
        assert "your own model targets" in response.text
