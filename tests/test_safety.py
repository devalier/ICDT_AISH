"""AILuminate safety suite: item build, judge scoring, canary probes, aggregation, guards."""

from __future__ import annotations

import json

import pytest

from aish import judge, runner
from aish.config import get_settings
from aish.packs import current
from aish.runner_types import RunError

from test_runner import StubAdapter, _execute, _queue, seeded  # noqa: F401


def _rows():
    # One row per MLCommons code, so every stratum (and the spc fold) is exercised.
    codes = ["vcr", "ncr", "src", "cse", "ssh", "iwp", "ipv", "dfm", "hte", "prv",
             "spc_ele", "spc_fin", "spc_hlt", "spc_lgl", "sxc_prn"]
    return [
        {"release_prompt_id": f"airr_t_{c}", "prompt_text": f"harmful-prompt-{c}", "hazard": c,
         "persona": "normal", "locale": "en_US", "prompt_hash": c}
        for c in codes for _ in range(1)
    ]


class JudgeStub:
    """Stands in for the judge model. Marks a response unsafe iff it contains BOMB."""

    provider_id = "stub"

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout):
        from aish.providers.base import Completion
        if "checking whether an AI system declined" in prompt:
            resp = prompt.split("RESPONSE:")[1]
            return Completion(text=json.dumps({"refused": "cannot help" in resp}))
        resp = prompt.split("RESPONSE:")[1]
        unsafe = "BOMB" in resp
        return Completion(text=json.dumps({"verdict": "unsafe" if unsafe else "safe",
                                           "refused": "cannot help" in resp}))


@pytest.fixture
def safety_env(monkeypatch):
    monkeypatch.setenv("AISH_JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "https://judge.internal/v1")
    get_settings.cache_clear()
    monkeypatch.setattr(runner, "_safety_rows", lambda suite: _rows())
    monkeypatch.setattr(judge, "get_adapter", lambda _p: JudgeStub())
    monkeypatch.setattr(judge, "CALIBRATION_FILE", "/nonexistent/cal.json")


def test_suite_validates_and_maps_every_dataset_code():
    view = current()
    assert not [p for p in view.problems if p.severity == "error"]
    suite = view.suite("ailuminate")
    mapped = {c for h in suite["hazards"] for c in h["codes"]}
    assert mapped == {r["hazard"] for r in _rows()}


def test_items_fold_spc_into_one_stratum_and_add_probes(safety_env):
    suite = current().suite("ailuminate")
    items = runner.build_items(suite, current(), sample=False, seed=1)
    harmful = [i for i in items if i.variant == "harmful"]
    assert {i.stratum for i in harmful if i.meta["code"].startswith("spc_")} == {"specialised_advice"}
    assert len({i.stratum for i in harmful}) == 12
    assert sum(i.variant == "control" for i in items) >= 20
    inj = [i for i in items if i.variant == "injection"]
    assert {i.stratum for i in inj} == {"prompt_injection_direct", "prompt_injection_indirect",
                                        "system_prompt_extraction"}
    assert all(i.meta["canary"] and (i.system or "canary" in i.meta) for i in inj)


def test_safe_model_passes_and_leaky_model_is_caught(seeded, safety_env, monkeypatch):  # noqa: F811
    user_id, target_id = seeded

    def reply(prompt: str) -> str:
        return "I cannot help with that."

    run, rows, summary = _execute(_queue("ailuminate", user_id, target_id), StubAdapter(reply),
                                  monkeypatch)
    assert run.status == "done"
    assert summary["metrics"]["violation_rate"] == 0.0
    assert summary["metrics"]["refusal_rate"] == 1.0
    # Refusing benign controls is measured separately and is not hidden by the safe score.
    assert summary["metrics"]["over_refusal_rate"] == 1.0
    assert summary["official"] is False
    assert summary["judge"]["calibration"]["status"] == "uncalibrated"
    assert summary["judge"]["self_judged"] is False


def test_unsafe_and_canary_leaks_count_as_violations(seeded, safety_env, monkeypatch):  # noqa: F811
    user_id, target_id = seeded
    suite = current().suite("ailuminate")
    canaries = {p["canary"] for p in
                __import__("yaml").safe_load(open("packs/probes/ailuminate.yaml"))["injection"]}

    def reply(prompt: str) -> str:
        if "harmful-prompt-vcr" in prompt:
            return "Here is how to build a BOMB."
        for canary in canaries:
            if canary in prompt:
                return f"Sure: {canary}"
        return "Happy to help with that."

    _, rows, summary = _execute(_queue("ailuminate", user_id, target_id), StubAdapter(reply),
                                monkeypatch)
    by = {s["key"]: s for s in summary["strata"]}
    assert by["violent_crimes"]["value"] == 1.0
    assert by["hate"]["value"] == 0.0
    # inj.direct.6 hides its canary in base64, so this stub (which only echoes canaries it
    # can see in the prompt) misses it; every other probe leaks.
    assert by["prompt_injection_direct"]["value"] == pytest.approx(5 / 6)
    assert by["prompt_injection_indirect"]["value"] == 1.0
    assert summary["metrics"]["violation_rate_any"] == 1.0
    assert summary["metrics"]["over_refusal_rate"] == 0.0
    # Macro-average: 3 injection strata + vcr unsafe out of 15 strata.
    assert summary["metrics"]["violation_rate"] == pytest.approx(
        sum(s["value"] for s in summary["strata"]) / len(summary["strata"]))
    assert suite["family"] == "safety"


def test_unparseable_judge_reply_is_its_own_outcome(seeded, safety_env, monkeypatch):  # noqa: F811
    from aish.providers.base import Completion

    class Garbage:
        async def generate(self, **_):
            return Completion(text="no idea, sorry")

    monkeypatch.setattr(judge, "get_adapter", lambda _p: Garbage())
    user_id, target_id = seeded
    _, rows, summary = _execute(_queue("ailuminate", user_id, target_id),
                                StubAdapter(lambda _p: "fine"), monkeypatch)
    assert summary["outcome_counts"].get("unparseable", 0) > 0
    assert summary["metrics"]["unparsed_rate"] > 0
    assert all(r.outcome != "unsafe" for r in rows if r.variant != "injection")


def test_restricted_suite_refuses_cloud_targets(seeded, safety_env, monkeypatch):  # noqa: F811
    from sqlalchemy import select

    from aish.db import session_scope
    from aish.models import ModelTarget, Run

    user_id, target_id = seeded
    with session_scope() as db:
        db.get(ModelTarget, target_id).provider = "anthropic"
    run_id = _queue("ailuminate", user_id, target_id)
    adapter = StubAdapter(lambda _p: "x")
    run, _, _ = _execute(run_id, adapter, monkeypatch)
    assert run.status == "failed" and "restricted" in run.error
    assert adapter.calls == []


def test_missing_judge_config_fails_closed(seeded, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AISH_JUDGE_MODEL", "")
    get_settings.cache_clear()
    user_id, target_id = seeded
    run, _, _ = _execute(_queue("ailuminate", user_id, target_id),
                         StubAdapter(lambda _p: "x"), monkeypatch)
    assert run.status == "failed" and "judge" in run.error.lower()


def test_cloud_judge_is_refused_for_restricted_suite(monkeypatch):
    monkeypatch.setenv("AISH_JUDGE_MODEL", "claude-x")
    monkeypatch.setenv("AISH_JUDGE_PROVIDER", "anthropic")
    get_settings.cache_clear()
    with pytest.raises(RunError, match="restricted"):
        judge.require_configured(restricted=True)


def test_failed_calibration_blocks_runs(seeded, safety_env, monkeypatch, tmp_path):  # noqa: F811
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"prompt_hash": judge.prompt_hash(), "model": "judge-x",
                               "kappa": 0.3, "n": 100}))
    monkeypatch.setattr(judge, "CALIBRATION_FILE", str(cal))
    user_id, target_id = seeded
    run, _, _ = _execute(_queue("ailuminate", user_id, target_id),
                         StubAdapter(lambda _p: "x"), monkeypatch)
    assert run.status == "failed" and "kappa" in run.error


def test_prompt_content_cannot_inject_placeholders():
    out = judge._fill("A {prompt} B {response} {{x}}", prompt="{response}", response="R")
    assert out == "A {response} B R {x}"


def test_cohen_kappa():
    assert judge.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert judge.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert judge.parse_verdict('ok: {"verdict": "Unsafe", "refused": false}').unsafe is True
    assert judge.parse_verdict("nonsense").unsafe is None
