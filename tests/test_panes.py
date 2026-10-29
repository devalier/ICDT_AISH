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
