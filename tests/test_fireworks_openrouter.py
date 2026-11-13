"""Fireworks AI and OpenRouter: bring-your-own-key model listing, chat calls, and the guard rails
(cloud classification, fixed hosts, data-collection policy). No real network: httpx is mocked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from aish.config import get_settings
from aish.providers import catalog, openai_adapter
from aish.providers.base import CLOUD_PROVIDERS, PROVIDERS, GenerationParams, ProviderError, get_adapter
from conftest import csrf_of

FIXTURE = Path(__file__).parent / "fixtures" / "openrouter_models_sample.json"


@pytest.fixture
def mock_http(monkeypatch):
    """Route every request from the catalog and adapters through a handler the test sets."""
    calls: list[httpx.Request] = []
    state = {"handler": None}

    def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return state["handler"](request)

    transport = httpx.MockTransport(dispatch)
    monkeypatch.setattr(catalog, "_TRANSPORT", transport)
    monkeypatch.setattr(openai_adapter, "_TRANSPORT", transport)
    return state, calls


def run(coro):
    return asyncio.run(coro)


def chat_reply(text="RECOMMEND", finish="stop", **extra):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": finish, **extra}],
                                     "usage": {"prompt_tokens": 11, "completion_tokens": 3}})


def generate(provider, key="sk-test", **kw):
    return run(get_adapter(provider).generate(
        model=kw.get("model", "m"), prompt="Hello", system=kw.get("system"), api_key=key,
        endpoint=kw.get("endpoint"), params=GenerationParams(max_tokens=64, seed=7), timeout=5))


# ---- registry and guard rails ---------------------------------------------------------- #


def test_both_are_registered_as_cloud_bring_your_own_key_providers():
    for pid in ("fireworks", "openrouter"):
        spec = PROVIDERS[pid]
        assert spec["requires_key"] and not spec["requires_endpoint"] and spec["cloud"]
        assert spec["endpoint"] == "none"
        assert pid in CLOUD_PROVIDERS
    from aish import judge
    assert judge.CLOUD_PROVIDERS is CLOUD_PROVIDERS            # one list for the whole app
    assert "openai_compatible" not in CLOUD_PROVIDERS


def test_a_key_is_required_to_call_either(mock_http):
    for pid in ("fireworks", "openrouter"):
        with pytest.raises(ProviderError, match="API key is required"):
            generate(pid, key=None)


# ---- chat calls -------------------------------------------------------------------------- #


def test_fireworks_calls_its_own_fixed_api_with_any_model_id(mock_http):
    state, calls = mock_http
    state["handler"] = lambda r: chat_reply("B")
    out = generate("fireworks", model="accounts/fireworks/models/llama-v3p3-70b-instruct",
                   endpoint="https://evil.example/v1", system="Be brief.")
    req = calls[-1]
    assert str(req.url) == "https://api.fireworks.ai/inference/v1/chat/completions"   # endpoint ignored
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "accounts/fireworks/models/llama-v3p3-70b-instruct"
    assert body["messages"][0] == {"role": "system", "content": "Be brief."} and body["max_tokens"] == 64
    assert out.text == "B" and out.tokens_in == 11 and not out.refused


def test_openrouter_sends_attribution_and_no_data_collection_by_default(mock_http):
    state, calls = mock_http
    state["handler"] = lambda r: chat_reply("Hi")
    generate("openrouter", model="meta-llama/llama-3.3-70b-instruct", endpoint="https://evil.example/v1")
    req = calls[-1]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    body = json.loads(req.content)
    assert body["provider"] == {"data_collection": "deny"} and body["seed"] == 7
    assert req.headers["x-title"] and req.headers["http-referer"].startswith("https://")


def test_openrouter_data_collection_can_be_allowed_by_the_deployment(mock_http, monkeypatch):
    monkeypatch.setenv("AISH_OPENROUTER_DATA_COLLECTION", "allow")
    get_settings.cache_clear()
    state, calls = mock_http
    state["handler"] = lambda r: chat_reply()
    generate("openrouter")
    assert json.loads(calls[-1].content)["provider"] == {"data_collection": "allow"}


def test_openrouter_moderation_block_is_a_refusal_not_an_error(mock_http):
    state, _ = mock_http
    state["handler"] = lambda r: httpx.Response(403, json={"error": {"message": "Input was flagged by moderation", "code": 403}})
    out = generate("openrouter")
    assert out.refused and out.text == "" and out.stop_reason == "moderation"


def test_insufficient_credit_and_upstream_errors_are_explained(mock_http):
    state, _ = mock_http
    state["handler"] = lambda r: httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
    with pytest.raises(ProviderError, match="insufficient credit"):
        generate("openrouter")
    state["handler"] = lambda r: chat_reply("", finish="error", error={"message": "upstream timed out"})
    with pytest.raises(ProviderError, match="upstream model error"):
        generate("openrouter")
    state["handler"] = lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})
    with pytest.raises(ProviderError, match="authentication rejected"):
        generate("fireworks")


@pytest.mark.parametrize("raw,answer", [
    ("<think>A or B? I think B.</think>\nThe answer is C", "The answer is C"),
    ("<think>still thinking about A", ""),                                  # cut off: no answer
    ("The answer is D", "The answer is D"),
])
def test_reasoning_blocks_are_not_scored_as_the_answer(mock_http, raw, answer):
    state, _ = mock_http
    state["handler"] = lambda r: chat_reply(raw)
    assert generate("fireworks").text == answer


# ---- model listing ------------------------------------------------------------------------ #


def test_openrouter_lists_chat_models_with_labels_without_needing_a_key(mock_http):
    state, calls = mock_http
    data = json.loads(FIXTURE.read_text())
    data["data"].append({"id": "acme/image-gen", "name": "Acme Image", "architecture": {"output_modalities": ["image"]},
                         "endpoints": ["/api/v1/chat/completions"]})
    state["handler"] = lambda r: httpx.Response(200, json=data)
    entries = run(catalog.list_model_entries("openrouter", None, None))
    ids = [e["id"] for e in entries]
    assert "meta-llama/llama-3.3-70b-instruct" in ids and "mistralai/mistral-small-3.2-24b-instruct" in ids
    assert not any(i.endswith(":batch") for i in ids)                     # batch-only, not chat
    assert "acme/image-gen" not in ids                                     # cannot answer a prompt
    assert ids == sorted(ids) and "authorization" not in calls[-1].headers
    llama = next(e for e in entries if e["id"] == "meta-llama/llama-3.3-70b-instruct")
    assert "Llama 3.3 70B" in llama["label"] and "context" in llama["label"] and "per 1M tokens" in llama["label"]
    free = next(e for e in entries if e["id"].endswith(":free"))
    assert "free" in free["label"]


def test_openrouter_sends_the_key_when_one_is_given(mock_http):
    state, calls = mock_http
    state["handler"] = lambda r: httpx.Response(200, json={"data": []})
    run(catalog.list_model_entries("openrouter", "sk-or-1", None))
    assert calls[-1].headers["authorization"] == "Bearer sk-or-1"


def _fw_model(name, **kw):
    return {"name": f"accounts/fireworks/models/{name}", "displayName": name.title(), "state": "READY",
            "kind": "HF_BASE_MODEL", "supportsServerless": True, "contextLength": 131072, **kw}


def test_fireworks_lists_serverless_chat_models_across_pages(mock_http):
    state, calls = mock_http
    pages = {
        None: {"models": [_fw_model("llama-v3p3-70b-instruct"), _fw_model("qwen3-235b", supportsImageInput=True),
                          _fw_model("needs-deployment", supportsServerless=False)], "nextPageToken": "p2"},
        "p2": {"models": [_fw_model("nomic-embed", kind="EMBEDDING_MODEL"),
                          _fw_model("old-model", deprecationDate={"year": 2027, "month": 1, "day": 15}),
                          _fw_model("uploading", state="UPLOADING")]},
    }

    def handler(request):
        assert request.headers["authorization"] == "Bearer fw-key"
        assert str(request.url).startswith("https://api.fireworks.ai/v1/accounts/fireworks/models")
        return httpx.Response(200, json=pages[request.url.params.get("pageToken")])

    state["handler"] = handler
    entries = run(catalog.list_model_entries("fireworks", "fw-key", "https://ignored.example"))
    ids = [e["id"].rsplit("/", 1)[-1] for e in entries]
    assert ids == ["llama-v3p3-70b-instruct", "old-model", "qwen3-235b"]
    labels = {e["id"].rsplit("/", 1)[-1]: e["label"] for e in entries}
    assert "131k context" in labels["llama-v3p3-70b-instruct"] and "vision" in labels["qwen3-235b"]
    assert "retiring 2027-01-15" in labels["old-model"]
    assert all(e["id"].startswith("accounts/fireworks/models/") for e in entries)   # usable as-is
    assert len(calls) == 2 and calls[0].url.params.get("pageSize") == "200"


def test_fireworks_falls_back_when_the_filter_is_not_accepted(mock_http):
    state, calls = mock_http

    def handler(request):
        if "filter" in request.url.params:
            return httpx.Response(400, json={"error": "bad filter"})
        return httpx.Response(200, json={"models": [_fw_model("a"), _fw_model("b", supportsServerless=False)]})

    state["handler"] = handler
    entries = run(catalog.list_model_entries("fireworks", "fw-key", None))
    assert [e["id"].rsplit("/", 1)[-1] for e in entries] == ["a"]


def test_fireworks_needs_a_key_to_list_and_reports_a_bad_one(mock_http):
    state, _ = mock_http
    with pytest.raises(ProviderError, match="API key is required"):
        run(catalog.list_model_entries("fireworks", None, None))
    state["handler"] = lambda r: httpx.Response(401, json={"error": {"message": "nope"}})
    with pytest.raises(ProviderError, match="rejected the API key"):
        run(catalog.list_model_entries("fireworks", "bad", None))


def test_an_unreachable_provider_is_reported_cleanly(mock_http):
    state, _ = mock_http

    def boom(request):
        raise httpx.ConnectError("down")

    state["handler"] = boom
    with pytest.raises(ProviderError, match="could not reach provider"):
        run(catalog.list_model_entries("openrouter", None, None))


# ---- the form ----------------------------------------------------------------------------- #


def test_the_form_offers_both_providers_with_their_notes(client, register):
    register(client)
    page = client.get("/models").text
    assert 'value="fireworks"' in page and 'value="openrouter"' in page
    assert 'data-endpoint="none"' in page and 'data-catalog="public"' in page
    assert "Load models from provider" in page and "data-endpoint-field" in page


def test_load_models_returns_ids_and_labels_for_the_users_key(client, register, mock_http):
    register(client)
    state, calls = mock_http
    state["handler"] = lambda r: httpx.Response(200, json=json.loads(FIXTURE.read_text()))
    r = client.post("/models/available", data={"csrf_token": csrf_of(client), "provider": "openrouter",
                                               "endpoint": "https://evil.example", "api_key": ""})
    assert r.status_code == 200
    body = r.json()
    assert "meta-llama/llama-3.3-70b-instruct" in body["models"]
    assert "Llama" in body["labels"]["meta-llama/llama-3.3-70b-instruct"]
    assert all(c.url.host == "openrouter.ai" for c in calls)               # the typed endpoint is never used
    r = client.post("/models/available", data={"csrf_token": csrf_of(client), "provider": "fireworks",
                                               "endpoint": "", "api_key": ""})
    assert r.status_code == 400 and "API key is required" in r.json()["error"]


def test_adding_a_model_stores_no_endpoint_and_encrypts_the_key(client, register):
    register(client)
    r = client.post("/models", data={"csrf_token": csrf_of(client), "label": "Llama via OpenRouter",
                                     "provider": "openrouter", "model_name": "meta-llama/llama-3.3-70b-instruct",
                                     "endpoint": "https://evil.example/v1", "api_key": "sk-or-v1-abcdef123456"},
                    follow_redirects=False)
    assert r.status_code == 303
    from aish.db import session_scope
    from aish.models import ModelTarget
    with session_scope() as db:
        t = db.query(ModelTarget).filter_by(label="Llama via OpenRouter").one()
        assert t.provider == "openrouter" and t.endpoint == "" and t.api_key_hint.endswith("3456")
        assert "sk-or-v1" not in (t.api_key_ciphertext or "")
    r = client.post("/models", data={"csrf_token": csrf_of(client), "label": "No key", "provider": "fireworks",
                                     "model_name": "accounts/fireworks/models/x", "api_key": ""})
    assert r.status_code == 400 and "requires an API key" in r.text


def test_restricted_suites_refuse_both_providers(client, register, monkeypatch):
    register(client)
    monkeypatch.setenv("AISH_JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "https://judge.internal/v1")
    get_settings.cache_clear()
    from aish.db import session_scope
    from aish.models import ModelTarget, Run
    with session_scope() as db:
        ids = []
        for pid in ("fireworks", "openrouter"):
            t = ModelTarget(user_id=1, label=pid, provider=pid, model_name="m")
            db.add(t); db.flush(); ids.append(t.id)
    for tid in ids:
        r = client.post("/runs", data={"csrf_token": csrf_of(client), "suite_id": "ailuminate",
                                       "target_ids": [str(tid)], "mode": "sample"})
        assert r.status_code == 409 and "Allow Safety Harness on Public Cloud Models" in r.text
    with session_scope() as db:
        assert db.query(Run).count() == 0
    from aish import judge
    monkeypatch.setenv("AISH_JUDGE_PROVIDER", "openrouter")
    get_settings.cache_clear()
    with pytest.raises(Exception, match="restricted"):
        judge.require_configured(restricted=True)
