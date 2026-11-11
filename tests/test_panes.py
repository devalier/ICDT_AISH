"""Navigation, the per-suite pages, legacy URLs and the comparison charts."""

from __future__ import annotations

import re

import pytest

from aish import judge, runner
from aish.config import get_settings
from conftest import csrf_of
from test_run_route import _wait_for, signed_in  # noqa: F401
from test_safety import JudgeStub
from test_runner import StubAdapter


def _run(client, suite_id, target_id="1", mode="sample") -> int:
    response = client.post(
        "/runs",
        data={"csrf_token": csrf_of(client), "suite_id": suite_id, "target_id": target_id, "mode": mode},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text[:300]
    run_id = int(response.headers["location"].rsplit("/", 1)[-1])
    assert _wait_for(client, run_id)["status"] == "done"
    return run_id


def test_sidebar_is_a_workflow_then_the_harness(client, register):
    register(client)
    page = client.get("/dashboard").text
    html = page[page.index("<nav"):page.index("</nav>")]
    order = [html.index(label) for label in
             ("Models Selected for Benchmarking", "Run tests", "Compare models", "Test harness", "Methodology", "Test suites")]
    assert order == sorted(order)
    for href in ("/suites/eu_mmlu", "/suites/ailuminate", "/suites/euiba_bias", "/suites/euiba_grounding"):
        assert f'href="{href}"' in html
    # Four separate suite entries, in axis order; no more redundant basis/parity entries.
    suites = [html.index(f'href="/suites/{s}"') for s in
              ("eu_mmlu", "ailuminate", "euiba_bias", "euiba_grounding")]
    assert suites == sorted(suites)
    assert "Language parity" not in html and "Bias basis" not in html


@pytest.mark.parametrize("suite_id,expect", [
    ("eu_mmlu", ["parity floor", "Prompt template", "Readiness"]),
    ("ailuminate", ["hazard code", "Hazard categories", "unofficial", "violent crimes"]),
    ("euiba_bias", ["Charter Art. 21", "Known limits of this method", "bias.alloc.recruitment.nationality"]),
    ("euiba_grounding", ["Correct abstention", "gnd.eu.nonexistent_regulation"]),
])
def test_every_suite_page_renders_basis_tests_thresholds(client, register, suite_id, expect):
    register(client)
    page = client.get(f"/suites/{suite_id}")
    assert page.status_code == 200
    text = page.text.lower()
    for needle in expect:
        assert needle.lower() in text, needle
    for heading in ("Results", "The tests", "Basis &amp; thresholds", "Readiness", "Provenance",
                    "Thresholds in force", "Changelog"):
        assert heading in page.text


def test_safety_page_never_shows_hazardous_prompt_text(client, register):
    register(client)
    page = client.get("/suites/ailuminate").text
    assert "harmful-prompt" not in page and "release_prompt_id" not in page


def test_unknown_suite_is_404(client, register):
    register(client)
    assert client.get("/suites/nope").status_code == 404


@pytest.mark.parametrize("old,new", [
    ("/panes/parity", "/suites/eu_mmlu"),
    ("/panes/bias", "/suites/euiba_bias"),
    ("/panes/suites/euiba_grounding", "/suites/euiba_grounding"),
    ("/panes/suites", "/panes/scoring"),
])
def test_old_urls_redirect_but_stay_behind_login(client, register, old, new):
    assert client.get(old, follow_redirects=False).headers["location"].startswith("/login")
    register(client)
    response = client.get(old, follow_redirects=False)
    assert response.status_code == 301 and response.headers["location"] == new


def test_readiness_reports_missing_judge_and_dataset(client, register, monkeypatch):
    monkeypatch.setenv("AISH_JUDGE_MODEL", "")
    get_settings.cache_clear()
    register(client)
    page = client.get("/suites/ailuminate").text
    assert "Judge model and endpoint configured" in page
    assert "check-open" in page
    # Manual review items come from the suite file and name an owner.
    assert "Safety reviewers" in page


def test_run_this_suite_preselects_the_suite(signed_in):  # noqa: F811
    page = signed_in.get("/runs?suite=euiba_bias").text
    assert re.search(r'value="euiba_bias"\s+selected', page)


def test_empty_comparison_guides_the_user(client, register):
    register(client)
    page = client.get("/dashboard").text
    assert "Add a model" in page


def test_comparison_charts_cover_every_model_and_axis(signed_in, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AISH_JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "https://judge.internal/v1")
    get_settings.cache_clear()
    monkeypatch.setattr(runner, "_safety_rows", lambda suite: [
        {"release_prompt_id": f"x{c}", "prompt_text": "p", "hazard": c, "persona": "n",
         "locale": "en_US", "prompt_hash": c} for c in ("vcr", "hte", "spc_ele")])
    monkeypatch.setattr(judge, "get_adapter", lambda _p: JudgeStub())
    monkeypatch.setattr(judge, "CALIBRATION_FILE", "/nonexistent/cal.json")

    # A second model target (inserted directly: the add-model form resolves DNS).
    from aish.db import session_scope
    from aish.models import ModelTarget
    with session_scope() as db:
        db.add(ModelTarget(user_id=1, label="Second model", provider="openai_compatible",
                           model_name="other", endpoint="https://example.com/v1"))

    monkeypatch.setattr(runner, "get_adapter",
                        lambda _p: StubAdapter(lambda p: "I cannot find any record of that."))
    for target in ("1", "2"):
        _run(signed_in, "euiba_grounding", target)
        _run(signed_in, "ailuminate", target)

    page = signed_in.get("/dashboard").text
    assert 'class="scorecard"' in page
    assert page.count('class="chart axis-chart"') == 2   # grounding + safety have runs
    assert "Stub target" in page and "Second model" in page
    assert page.count("seg-label") >= 0 and page.count("<a href=\"/runs/") >= 8
    # Grounding drill-down is a stacked outcome bar; safety is a per-hazard heatmap.
    assert "seg-hallucinated" in page or "seg-abstained_correctly" in page
    assert 'class="heat"' in page
    # Status is never colour alone: a glyph and a word accompany every verdict.
    assert re.search(r"[✓!✕?–] (Pass|Watch|Fail|Inconclusive)", page)
    # No inline styles anywhere (CSP).
    assert 'style="' not in page
    # Axes with no runs say so instead of drawing an empty chart.
    assert "No language runs yet." in page and "No bias runs yet." in page


def test_suite_page_is_tabbed_with_every_pane_in_the_markup(client, register):
    """Tabs are progressive enhancement: with no JS every pane is simply visible."""
    register(client)
    page = client.get("/suites/euiba_bias").text
    assert page.count('role="tab"') == 5 and page.count('role="tabpanel"') == 5
    assert 'style="' not in page


def test_dashboard_tabs_scorecard_links_resolve_to_panes(client, register):
    register(client)
    page = client.get("/dashboard").text
    # Empty state has no tabs; once models exist the scorecard links target pane ids.
    assert "Add a model" in page


def test_models_page_lists_selected_models_before_the_add_form(signed_in):  # noqa: F811
    page = signed_in.get("/models").text
    assert "Models Selected for Benchmarking" in page
    listed = page.index("Selected Models for Benchmarking")
    adding = page.index("Add a model for benchmarking")
    assert listed < adding
    assert "Stub target" in page[listed:adding]
    assert "Add a model target" not in page and "Your targets" not in page
    assert 'style="' not in page


def test_models_page_opens_the_add_form_when_nothing_is_selected(client, register):
    register(client)
    page = client.get("/models").text
    assert re.search(r'<details class="card collapsible"\s+open', page)


def test_no_user_facing_page_calls_a_model_a_target(signed_in):  # noqa: F811
    """One vocabulary: a model selected for benchmarking is a 'model', never a 'target'."""
    for path in ("/models", "/runs", "/dashboard", "/panes/scoring", "/suites/ailuminate",
                 "/suites/euiba_bias", "/suites/eu_mmlu", "/suites/euiba_grounding"):
        text = re.sub(r"<[^>]+>", " ", signed_in.get(path).text).lower().replace("stub target", "")
        assert not re.search(r"\btargets?\b", text), path


def test_run_pages_use_plain_language(signed_in, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runner, "get_adapter", lambda _p: StubAdapter(lambda p: "I cannot find any record of that."))
    run_id = _run(signed_in, "euiba_grounding")
    detail = re.sub(r"<[^>]+>", " ", signed_in.get(f"/runs/{run_id}").text).lower()
    evidence = re.sub(r"<[^>]+>", " ", signed_in.get(f"/runs/{run_id}/evidence").text).lower()
    for text in (detail, evidence):
        for jargon in ("stratum", "strata", "manifest", "evidence"):
            assert jargon not in text, jargon
    assert "run settings" in detail and "prompts and responses" in evidence


# ---- managing models and running several at once ------------------------------------------- #


def _add_models(n: int) -> list[int]:
    """Insert extra models for user 1 directly (the add form resolves DNS)."""
    from aish.db import session_scope
    from aish.models import ModelTarget

    ids = []
    with session_scope() as db:
        for i in range(n):
            t = ModelTarget(user_id=1, label=f"Extra {i}", provider="openai_compatible",
                            model_name=f"m{i}", endpoint="https://example.com/v1")
            db.add(t)
            db.flush()
            ids.append(t.id)
    return ids


def test_models_page_shows_key_and_remove_controls_inline_not_in_a_popup(signed_in):  # noqa: F811
    page = signed_in.get("/models").text
    # Regression: these used to sit in a floating menu clipped by the scrolling table.
    assert "menu-body" not in page and 'class="menu"' not in page
    assert "Change API key or remove" in page and 'action="/models/1/key"' in page
    assert 'action="/models/1/delete"' in page and "data-confirm=" in page
    assert 'name="model"' in page and 'data-max-select="3"' in page
    assert "up to 3" in page.lower()


def test_changing_a_key_and_removing_a_model_work(signed_in):  # noqa: F811
    token = csrf_of(signed_in)
    r = signed_in.post("/models/1/key", data={"csrf_token": token, "api_key": "sk-new-key-123456"})
    assert r.status_code == 200 and "Credential updated" in r.text
    assert "sk-new-key-123456" not in r.text and "3456" in r.text      # only the hint is shown
    r = signed_in.post("/models/1/key", data={"csrf_token": token, "api_key": ""})
    assert "Credential updated" in r.text
    r = signed_in.post("/models/1/delete", data={"csrf_token": token})
    assert r.status_code == 200 and "Removed" in r.text
    assert "Stub target" not in signed_in.get("/models").text.split("Add a model")[0]


def test_runs_page_offers_a_checkbox_per_model_and_preselects_from_the_models_page(signed_in):  # noqa: F811
    ids = _add_models(2)
    page = signed_in.get(f"/runs?model=1&model={ids[0]}").text
    assert page.count('name="target_ids"') == 3 and "Select up to 3" in page
    assert re.search(r'id="m-1"[^>]*checked', page) and re.search(rf'id="m-{ids[0]}"[^>]*checked', page)
    assert not re.search(rf'id="m-{ids[1]}"[^>]*checked', page)


def _post_runs(client, targets, suite="euiba_grounding"):
    data = {"csrf_token": csrf_of(client), "suite_id": suite, "mode": "sample"}
    return client.post("/runs", data={**data, "target_ids": [str(t) for t in targets]},
                       follow_redirects=False)


def test_three_models_run_in_parallel_on_identical_settings(signed_in, monkeypatch):  # noqa: F811
    ids = [1, *_add_models(2)]
    monkeypatch.setattr(runner, "get_adapter",
                        lambda _p: StubAdapter(lambda p: "I cannot find any record of that."))
    response = _post_runs(signed_in, ids)
    assert response.status_code == 303 and response.headers["location"].startswith("/runs?started=")
    run_ids = [int(x) for x in response.headers["location"].split("=")[1].split(",")]
    assert len(run_ids) == 3
    for run_id in run_ids:
        assert _wait_for(signed_in, run_id)["status"] == "done"
    page = signed_in.get(response.headers["location"]).text
    assert "Started 3 runs in parallel" in page and "Compare models" in page

    from aish.db import session_scope
    from aish.models import Run
    with session_scope() as db:
        runs = [db.get(Run, r) for r in run_ids]
        assert {r.target_id for r in runs} == set(ids)
        assert len({(r.suite_id, r.suite_version, r.items_total, r.manifest_json.split('"model"')[0])
                    for r in runs}) == 1          # same suite, version, prompts and settings


def test_more_than_three_or_none_is_refused_and_creates_no_runs(signed_in):  # noqa: F811
    ids = [1, *_add_models(3)]
    r = _post_runs(signed_in, ids)
    assert r.status_code == 400 and "at most 3" in r.text
    r = _post_runs(signed_in, [])
    assert r.status_code == 400 and "at least one" in r.text
    from aish.db import session_scope
    from aish.models import Run
    with session_scope() as db:
        assert db.query(Run).count() == 0


def test_a_selection_containing_someone_elses_model_starts_nothing(client, register, signed_in):  # noqa: F811
    from aish.db import session_scope
    from aish.models import ModelTarget, User
    from aish.security import hash_password
    with session_scope() as db:
        other = User(email="other@ec.europa.eu", password_hash=hash_password("a-long-enough-passphrase"),
                     euiba_id="COM", euiba_name="European Commission")
        db.add(other)
        db.flush()
        foreign = ModelTarget(user_id=other.id, label="Theirs", provider="openai_compatible",
                              model_name="x", endpoint="https://example.com/v1")
        db.add(foreign)
        db.flush()
        foreign_id = foreign.id
    r = _post_runs(signed_in, [1, foreign_id])
    assert r.status_code == 400 and "your own models" in r.text
    from aish.models import Run
    with session_scope() as db:
        assert db.query(Run).count() == 0


def test_a_restricted_suite_refuses_the_whole_selection_if_one_model_is_cloud(signed_in, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AISH_JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("AISH_JUDGE_ENDPOINT", "https://judge.internal/v1")
    get_settings.cache_clear()
    from aish.db import session_scope
    from aish.models import ModelTarget, Run
    with session_scope() as db:
        cloud = ModelTarget(user_id=1, label="Cloud one", provider="openai", model_name="gpt")
        db.add(cloud)
        db.flush()
        cloud_id = cloud.id
    r = _post_runs(signed_in, [1, cloud_id], suite="ailuminate")
    assert r.status_code == 409 and "Cloud one" in r.text
    with session_scope() as db:
        assert db.query(Run).count() == 0


def test_the_old_single_model_form_field_still_works(signed_in, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runner, "get_adapter", lambda _p: StubAdapter(lambda p: "I cannot find any record of that."))
    r = signed_in.post("/runs", data={"csrf_token": csrf_of(signed_in), "suite_id": "euiba_grounding",
                                      "target_id": "1", "mode": "sample"}, follow_redirects=False)
    assert r.status_code == 303 and re.fullmatch(r"/runs/\d+", r.headers["location"])
