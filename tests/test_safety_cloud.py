"""The Safety (restricted) suite on public cloud models: the user's switch, the run-page guidance
and confirmation, enforcement at run time, and the consent screen before customising."""

from __future__ import annotations

import json
import re

import pytest

from aish import judge, prefs, runner
from aish.config import get_settings
from conftest import csrf_of
from test_run_route import _wait_for, signed_in  # noqa: F401
from test_runner import StubAdapter
from test_safety import JudgeStub, _rows

SWITCH = "Allow Safety Harness on Public Cloud Models"


@pytest.fixture
def safety_ready(monkeypatch):
    monkeypatch.setenv("AISH_JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "https://judge.internal/v1")
    get_settings.cache_clear()
    monkeypatch.setattr(runner, "_safety_rows", lambda suite: _rows())
    monkeypatch.setattr(judge, "get_adapter", lambda _p: JudgeStub())
    monkeypatch.setattr(judge, "CALIBRATION_FILE", "/nonexistent/cal.json")
    monkeypatch.setattr(runner, "get_adapter", lambda _p: StubAdapter(lambda q: "I cannot help with that."))


def _cloud_model():
    from aish.db import session_scope
    from aish.models import ModelTarget
    with session_scope() as db:
        t = ModelTarget(user_id=1, label="GPT in the cloud", provider="openai", model_name="gpt-x")
        db.add(t)
        db.flush()
        return t.id


def _switch(client, on: bool):
    data = {"csrf_token": csrf_of(client)}
    if on:
        data["allowed"] = "1"
    return client.post("/suites/ailuminate/cloud-permission", data=data, follow_redirects=False)


def _run(client, target_ids, confirm=False):
    data = {"csrf_token": csrf_of(client), "suite_id": "ailuminate", "mode": "sample",
            "target_ids": [str(t) for t in target_ids]}
    if confirm:
        data["confirm_cloud_safety"] = "1"
    return client.post("/runs", data=data, follow_redirects=False)


# ---- the switch ------------------------------------------------------------------------ #


def test_the_safety_page_has_the_switch_and_the_red_warning(signed_in):  # noqa: F811
    page = signed_in.get("/suites/ailuminate").text
    assert SWITCH in page and 'role="switch"' in page and "danger-text" in page
    assert "API revocation or account suspension" in page and "certain about its purpose and function" in page
    assert re.search(r'name="allowed"[^>]*role="switch"\s+data-confirm-on', page)   # starts off (not checked)
    assert SWITCH not in signed_in.get("/suites/euiba_bias").text                       # only on restricted suites


def test_switching_on_and_off_is_saved_per_user_and_audited(signed_in):  # noqa: F811
    r = _switch(signed_in, True)
    assert r.status_code == 303 and "cloud=on" in r.headers["location"]
    page = signed_in.get(r.headers["location"]).text
    assert "switched on" in page and re.search(r'value="1" role="switch"\s+checked', page)
    _switch(signed_in, False)
    from aish.db import session_scope
    from aish.models import AuditEvent
    with session_scope() as db:
        assert prefs.cloud_allowed(db, 1) is False
        events = [e.event for e in db.query(AuditEvent).order_by(AuditEvent.id)]
    assert "safety.cloud_allowed" in events and "safety.cloud_disallowed" in events


def test_a_deployment_can_lock_the_switch_off(signed_in, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AISH_ALLOW_SAFETY_ON_CLOUD_TOGGLE", "false")
    get_settings.cache_clear()
    _switch(signed_in, True)
    from aish.db import session_scope
    with session_scope() as db:
        assert prefs.cloud_allowed(db, 1) is False
    page = signed_in.get("/suites/ailuminate").text
    assert "Locked off" in page and "disabled" in page


def test_the_switch_requires_a_csrf_token(signed_in):  # noqa: F811
    r = signed_in.post("/suites/ailuminate/cloud-permission", data={"allowed": "1"})
    assert r.status_code in (400, 403)


# ---- run page guidance ------------------------------------------------------------------- #


def test_run_page_explains_how_to_allow_cloud_when_off(signed_in):  # noqa: F811
    _cloud_model()
    page = signed_in.get("/runs").text
    assert "data-safety-panel" in page and 'data-restricted="yes"' in page
    assert "blocked" in page and SWITCH in page and "/suites/ailuminate#cloud-permission" in page
    assert 'data-cloud="yes"' in page and "public cloud" in page and "on-prem" in page
    assert 'name="confirm_cloud_safety"' not in page


def test_run_page_asks_for_confirmation_when_on(signed_in):  # noqa: F811
    _switch(signed_in, True)
    page = signed_in.get("/runs").text
    assert "is <strong>on</strong>" in page and 'name="confirm_cloud_safety"' in page
    assert "revoke my API key or suspend my account" in page


# ---- enforcement --------------------------------------------------------------------------- #


def test_cloud_model_refused_while_off_with_instructions(signed_in, safety_ready):  # noqa: F811
    cloud = _cloud_model()
    r = _run(signed_in, [cloud], confirm=True)
    assert r.status_code == 409 and SWITCH in r.text and "Safety test suite page" in r.text


def test_cloud_model_needs_the_per_run_confirmation_when_on(signed_in, safety_ready):  # noqa: F811
    cloud = _cloud_model()
    _switch(signed_in, True)
    r = _run(signed_in, [cloud])
    assert r.status_code == 409 and "Tick the box" in r.text
    from aish.db import session_scope
    from aish.models import Run
    with session_scope() as db:
        assert db.query(Run).count() == 0


def test_cloud_model_runs_when_on_and_confirmed(signed_in, safety_ready):  # noqa: F811
    cloud = _cloud_model()
    _switch(signed_in, True)
    r = _run(signed_in, [cloud], confirm=True)
    assert r.status_code == 303, r.text[:400]
    run_id = int(r.headers["location"].rsplit("/", 1)[-1])
    assert _wait_for(signed_in, run_id)["status"] == "done"
    from aish.db import session_scope
    from aish.models import Run
    with session_scope() as db:
        assert json.loads(db.get(Run, run_id).manifest_json)["cloud_safety_confirmed"] is True


def test_on_prem_models_never_need_the_switch(signed_in, safety_ready):  # noqa: F811
    r = _run(signed_in, [1])
    assert r.status_code == 303
    assert _wait_for(signed_in, int(r.headers["location"].rsplit("/", 1)[-1]))["status"] == "done"


def test_switching_off_after_queueing_stops_the_run(signed_in, safety_ready, monkeypatch):  # noqa: F811
    cloud = _cloud_model()
    _switch(signed_in, True)
    monkeypatch.setattr("aish.web.runs.queue_run", lambda run_id: None)   # queue it but do not start it yet
    r = _run(signed_in, [cloud], confirm=True)
    run_id = int(r.headers["location"].rsplit("/", 1)[-1])
    _switch(signed_in, False)
    import asyncio
    asyncio.run(runner.execute_run(run_id))
    from aish.db import session_scope
    from aish.models import Run
    with session_scope() as db:
        run = db.get(Run, run_id)
        assert run.status == "failed" and SWITCH in run.error


def test_the_judge_rule_is_unchanged_a_cloud_judge_still_refused(monkeypatch):
    monkeypatch.setenv("AISH_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("AISH_JUDGE_MODEL", "x")
    get_settings.cache_clear()
    with pytest.raises(Exception, match="restricted"):
        judge.require_configured(restricted=True)


# ---- consent before customising -------------------------------------------------------------- #


def test_customising_safety_shows_the_consent_screen_first(signed_in):  # noqa: F811
    for path in ("/suites/ailuminate/customise", "/suites/ailuminate/customise/yaml"):
        page = signed_in.get(path).text
        assert "Before you customise" in page and "sensitive and potentially" in page
        assert "Only proceed if you understand the" in page and 'name="understood"' in page
        assert "data-yaml-editor" not in page and "details class=\"item" not in page
    # Other suites open straight away.
    assert "Before you customise" not in signed_in.get("/suites/euiba_bias/customise").text


def test_edits_are_refused_until_consent_is_given(signed_in):  # noqa: F811
    r = signed_in.post("/suites/ailuminate/customise/item", data={
        "csrf_token": csrf_of(signed_in), "section": "controls", "op": "add"})
    assert r.status_code == 403 and "Before you customise" in r.text
    r = signed_in.post("/suites/ailuminate/customise/draft", data={"csrf_token": csrf_of(signed_in), "action": "preview"})
    assert r.status_code == 403
    from aish.db import session_scope
    from aish.models import CustomDraft
    with session_scope() as db:
        assert db.query(CustomDraft).count() == 0


def test_consent_needs_the_box_ticked_then_opens_the_editor_and_is_logged(signed_in):  # noqa: F811
    r = signed_in.post("/suites/ailuminate/customise/consent", data={"csrf_token": csrf_of(signed_in)})
    assert r.status_code == 400 and "Before you customise" in r.text
    r = signed_in.post("/suites/ailuminate/customise/consent",
                       data={"csrf_token": csrf_of(signed_in), "understood": "yes"}, follow_redirects=False)
    assert r.status_code == 303
    page = signed_in.get("/suites/ailuminate/customise").text
    assert "Before you customise" not in page and "ctl.vcr.1" in page
    from aish.db import session_scope
    from aish.models import AuditEvent
    with session_scope() as db:
        assert db.query(AuditEvent).filter_by(event="safety.content_acknowledged").count() == 1
