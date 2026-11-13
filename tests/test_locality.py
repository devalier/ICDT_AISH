"""Where a model really is: an "On-prem / OpenAI-compatible" URL on the public internet is
public cloud, so the Safety rules and the on-prem-judge rule follow it."""

from __future__ import annotations

import socket

import pytest

from aish import judge
from aish.config import get_settings
from aish.providers import locality
from aish.providers.locality import dedicated_provider, is_cloud_target
from aish.runner_types import RunError
from conftest import csrf_of
from test_run_route import signed_in  # noqa: F401


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    table = {"llm.example.org": "93.184.216.34", "vllm.euiba.internal": "10.1.2.3",
             "gpu.euiba.eu": "62.40.1.1"}

    def resolve(host, port, *a, **k):
        if host not in table:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]

    monkeypatch.setattr(locality, "_lookup", resolve)
    locality._cache.clear()
    yield
    locality._cache.clear()
    get_settings.cache_clear()


@pytest.mark.parametrize("provider,endpoint,cloud", [
    ("openai", None, True),
    ("fireworks", None, True),
    ("openai_compatible", "https://api.fireworks.ai/inference/v1", True),
    ("openai_compatible", "https://openrouter.ai/api/v1", True),
    ("openai_compatible", "https://foo.openai.azure.com/openai", True),
    ("openai_compatible", "https://llm.example.org/v1", True),        # public address
    ("openai_compatible", "http://93.184.216.34:8000/v1", True),      # public IP literal
    ("openai_compatible", "http://vllm.euiba.internal:8000/v1", False),  # private address
    ("openai_compatible", "http://10.0.0.5:8000/v1", False),
    ("openai_compatible", "http://unknown.corp/v1", False),           # unresolved: internal name
    ("openai_compatible", "", False),
])
def test_classification(provider, endpoint, cloud):
    assert is_cloud_target(provider, endpoint) is cloud


def test_onprem_hosts_exempt_an_institutions_public_range_but_never_a_known_cloud(monkeypatch):
    assert is_cloud_target("openai_compatible", "https://gpu.euiba.eu/v1")
    monkeypatch.setenv("AISH_ONPREM_HOSTS", "euiba.eu, fireworks.ai")
    get_settings.cache_clear()
    assert not is_cloud_target("openai_compatible", "https://gpu.euiba.eu/v1")
    assert is_cloud_target("openai_compatible", "https://api.fireworks.ai/inference/v1")


def test_dedicated_provider_is_suggested():
    assert dedicated_provider("openai_compatible", "https://api.fireworks.ai/inference/v1") == "fireworks"
    assert dedicated_provider("openai_compatible", "https://openrouter.ai/api/v1") == "openrouter"
    assert dedicated_provider("openai_compatible", "http://10.0.0.5/v1") is None
    assert dedicated_provider("fireworks", None) is None


def test_a_judge_on_a_public_url_is_a_cloud_judge(monkeypatch):
    monkeypatch.setenv("AISH_JUDGE_MODEL", "j")
    monkeypatch.setenv("AISH_JUDGE_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "https://api.fireworks.ai/inference/v1")
    get_settings.cache_clear()
    with pytest.raises(RunError, match="cloud judge"):
        judge.require_configured(restricted=True)
    judge.require_configured(restricted=False)
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "http://vllm.euiba.internal:8000/v1")
    get_settings.cache_clear()
    judge.require_configured(restricted=True)


def _add(client, endpoint):
    from aish.db import session_scope
    from aish.models import ModelTarget
    with session_scope() as db:
        t = ModelTarget(user_id=1, label="Fireworks as on-prem", provider="openai_compatible",
                        model_name="accounts/fireworks/models/llama", endpoint=endpoint)
        db.add(t)
        db.flush()
        return t.id


def test_safety_refuses_a_public_url_listed_as_on_prem(signed_in):  # noqa: F811
    tid = _add(signed_in, "https://api.fireworks.ai/inference/v1")
    response = signed_in.post("/runs", data={
        "csrf_token": csrf_of(signed_in), "suite_id": "ailuminate", "mode": "sample",
        "target_ids": [str(tid)]}, follow_redirects=False)
    assert response.status_code == 409
    assert "Fireworks as on-prem" in response.text
    page = signed_in.get("/runs").text
    assert 'data-cloud="yes"' in page
    assert "public cloud" in signed_in.get("/models").text


def test_loading_models_from_a_known_service_as_on_prem_says_which_provider_to_use(signed_in, monkeypatch):  # noqa: F811
    from aish.web import targets

    async def boom(*a, **k):
        raise RuntimeError("html instead of json")

    monkeypatch.setattr(targets, "list_model_entries", boom)
    response = signed_in.post("/models/available", data={
        "csrf_token": csrf_of(signed_in), "provider": "openai_compatible",
        "endpoint": "https://api.fireworks.ai/inference/v1", "api_key": "fw-x"})
    assert response.status_code == 502
    body = response.json()
    assert "Fireworks AI" in body["error"] and "instead of on-prem" in body["error"]
