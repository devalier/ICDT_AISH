"""Run execution, scoring and aggregation, against a stubbed model."""

from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select

from aish import runner
from aish.db import session_scope
from aish.models import ModelTarget, Result, Run, User
from aish.packs import current
from aish.providers.base import Completion, ProviderError
from aish.security import encrypt_secret, hash_password


class StubAdapter:
    """Replaces a real provider. Replies are driven by a callable so a test can
    make the model behave any way it needs to."""

    provider_id = "stub"
    requires_endpoint = False
    requires_key = False

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[str] = []

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout):
        self.calls.append(prompt)
        value = self.reply(prompt)
        if isinstance(value, Exception):
            raise value
        return Completion(text=value, tokens_in=10, tokens_out=5, latency_ms=7)


@pytest.fixture
def seeded():
    with session_scope() as db:
        user = User(
            email="runner@ec.europa.eu",
            password_hash=hash_password("a-long-enough-passphrase"),
            euiba_id="COM",
            euiba_name="European Commission",
        )
        db.add(user)
        db.flush()
        target = ModelTarget(
            user_id=user.id,
            label="Stub",
            provider="openai_compatible",
            model_name="stub-model",
            endpoint="https://example.com/v1",
            api_key_ciphertext=encrypt_secret("sk-secret-value", f"user:{user.id}"),
            api_key_hint="****alue",
        )
        db.add(target)
        db.flush()
        return user.id, target.id


def _queue(suite_id: str, user_id: int, target_id: int, sample: bool = True) -> int:
    view = current()
    suite = view.suite(suite_id)
    with session_scope() as db:
        run = Run(
            user_id=user_id,
            target_id=target_id,
            suite_id=suite_id,
            suite_version=str(suite.get("version")),
            family=str(suite.get("family")),
            status="queued",
            sample_mode=sample,
            manifest_json=json.dumps({"seed": 20260921, "max_tokens": 256}),
        )
        db.add(run)
        db.flush()
        return run.id


def _execute(run_id: int, adapter, monkeypatch) -> tuple[Run, list[Result], dict]:
    monkeypatch.setattr(runner, "get_adapter", lambda _provider: adapter)
    asyncio.run(runner.execute_run(run_id))
    with session_scope() as db:
        run = db.get(Run, run_id)
        rows = list(db.scalars(select(Result).where(Result.run_id == run_id)))
        return run, rows, json.loads(run.summary_json or "{}")


# --------------------------------------------------------------------------- #


def test_grounding_run_scores_correct_abstention_positively(seeded, monkeypatch):
    user_id, target_id = seeded

    def reply(prompt: str) -> str:
        # Decline the unanswerable items, answer the answerable one.
        if "2027/1180" in prompt or "C-994/24" in prompt or "2024" in prompt:
            return "I cannot find any record of that. It does not exist."
        if "right of legislative initiative" in prompt:
            return "The European Commission."
        return "Paragraph 10 states that conciliation follows two readings."

    run, rows, summary = _execute(_queue("euiba_grounding", user_id, target_id),
                                  StubAdapter(reply), monkeypatch)

    assert run.status == "done"
    assert len(rows) == 5
    outcomes = {row.item_id: row.outcome for row in rows}
    assert outcomes["gnd.eu.nonexistent_regulation"] == "abstained_correctly"
    assert outcomes["gnd.eu.fabricated_case_law"] == "abstained_correctly"
    assert outcomes["gnd.eu.answerable_not_trick"] == "grounded"
    assert summary["metrics"]["hallucination_rate"] == 0.0
    assert summary["verdict"] in ("pass", "watch", "fail", "inconclusive")


def test_a_model_that_refuses_everything_is_not_rewarded(seeded, monkeypatch):
    """Blanket abstention must not look like good grounding — that is what the
    calibration metric exists to catch."""
    user_id, target_id = seeded
    adapter = StubAdapter(lambda _prompt: "I don't know.")
    _, rows, summary = _execute(_queue("euiba_grounding", user_id, target_id),
                                adapter, monkeypatch)

    outcomes = [row.outcome for row in rows]
    assert "abstained_wrongly" in outcomes
    assert summary["metrics"]["calibration"] < 1.0
    assert summary["metrics"]["grounded_rate"] == 0.0


def test_bias_run_detects_a_disparity_and_reports_per_variant(seeded, monkeypatch):
    user_id, target_id = seeded

    def reply(prompt: str) -> str:
        # A model that declines one nationality and recommends the rest.
        if "a RO national" in prompt:
            return "DO NOT RECOMMEND. Insufficient evidence."
        if "RECOMMEND" in prompt or "recruitment panel" in prompt:
            return "RECOMMEND. Strong profile."
        if "APPROVE" in prompt or "hardship grant" in prompt:
            return "APPROVE"
        return "A three-sentence summary of the engineer's performance this year."

    run, rows, summary = _execute(_queue("euiba_bias", user_id, target_id),
                                  StubAdapter(reply), monkeypatch)

    assert run.status == "done"
    assert summary["groups"], "expected per-probe counterfactual groups"
    recruitment = next(
        (g for g in summary["groups"] if g["probe_id"] == "bias.alloc.recruitment.nationality"),
        None,
    )
    if recruitment:  # the sampled run may not draw RO into the variant subset
        assert len(recruitment["variants"]) >= 2
        assert recruitment["ci"][0] <= recruitment["disparity"] + 1e-9
    assert "rate_disparity" in summary["metrics"]


def test_transport_failures_are_recorded_as_errors_not_wrong_answers(seeded, monkeypatch):
    user_id, target_id = seeded
    adapter = StubAdapter(lambda _p: ProviderError("could not reach model endpoint", retryable=True))
    run, rows, summary = _execute(_queue("euiba_grounding", user_id, target_id),
                                  adapter, monkeypatch)

    assert all(row.outcome == "error" for row in rows)
    assert all(row.score is None for row in rows)
    assert run.status == "failed"


def test_the_api_key_never_appears_in_a_stored_error(seeded, monkeypatch):
    user_id, target_id = seeded
    adapter = StubAdapter(lambda _p: ProviderError("upstream rejected key sk-secret-value"))
    _, rows, _ = _execute(_queue("euiba_grounding", user_id, target_id), adapter, monkeypatch)

    for row in rows:
        assert "sk-secret-value" not in row.error
        assert "****" in row.error


def test_unparseable_answers_are_their_own_category(seeded, monkeypatch):
    from aish.scoring import UNPARSEABLE, parse_choice

    assert parse_choice("I'd rather not say.", ["a", "b", "c", "d"]).method == UNPARSEABLE
    # And the runner maps that to an outcome with no score, so it cannot be counted
    # as a wrong answer.
    item = runner.Item(item_id="x", prompt="p", meta={"choices": ["a", "b", "c", "d"], "answer": 0})
    outcome, score = runner.score_item("capability", item, "I'd rather not say.", False)
    assert outcome == UNPARSEABLE
    assert score is None


def test_a_provider_refusal_is_its_own_outcome(seeded, monkeypatch):
    item = runner.Item(item_id="x", prompt="p", meta={})
    outcome, score = runner.score_item("grounding", item, "", refused=True)
    assert outcome == "refused"
    assert score is None


def test_manifest_records_everything_comparability_depends_on(seeded, monkeypatch):
    user_id, target_id = seeded
    run_id = _queue("euiba_grounding", user_id, target_id)
    _execute(run_id, StubAdapter(lambda _p: "answer"), monkeypatch)
    with session_scope() as db:
        manifest = json.loads(db.get(Run, run_id).manifest_json)
    assert manifest["seed"] == 20260921


def test_eu_mmlu_reports_a_clear_message_when_the_dataset_is_absent(seeded, monkeypatch):
    view = current()
    monkeypatch.setattr(runner, "EU_MMLU_DATA", runner.Path("data/definitely-not-here.jsonl"))
    with pytest.raises(runner.RunError) as exc:
        runner.build_items(view.suite("eu_mmlu"), view, sample=True, seed=1)
    assert "not provisioned" in str(exc.value)


def test_eu_mmlu_scores_and_computes_parity(seeded, monkeypatch, tmp_path):
    """Exercised against a small fixture; the real dataset is provisioned separately."""
    fixture = tmp_path / "eu_mmlu.jsonl"
    rows = []
    for language in ("en", "lt"):
        for index in range(4):
            rows.append(
                {
                    "Language": language, "Subject": "international_law", "Split": "test",
                    "Index": index, "Question": f"Question {index} in {language}?",
                    "Choice_0": "alpha", "Choice_1": "beta", "Choice_2": "gamma",
                    "Choice_3": "delta", "Answer": 0,
                }
            )
    fixture.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(runner, "EU_MMLU_DATA", fixture)

    user_id, target_id = seeded

    def reply(prompt: str) -> str:
        # Right in English, wrong in Lithuanian: a parity failure, not a knowledge one.
        return "The answer is A." if "in en?" in prompt else "The answer is B."

    _, rows_out, summary = _execute(_queue("eu_mmlu", user_id, target_id),
                                    StubAdapter(reply), monkeypatch)

    assert len(rows_out) == 8
    assert summary["language_accuracy"]["en"] == 1.0
    assert summary["language_accuracy"]["lt"] == 0.0
    assert summary["metrics"]["parity_floor"] == 0.0
    assert summary["verdicts"]["parity_floor"] == "fail"


def test_an_oversized_run_is_refused_rather_than_truncated(seeded, monkeypatch, tmp_path):
    """Slicing an ordered item list would drop whole languages from the tail and
    still report a parity floor over the survivors. Refusing is the honest outcome."""
    fixture = tmp_path / "eu_mmlu.jsonl"
    rows = [
        {
            "Language": language, "Subject": "international_law", "Split": "test",
            "Index": index, "Question": f"Q{index} in {language}?",
            "Choice_0": "a", "Choice_1": "b", "Choice_2": "c", "Choice_3": "d",
            "Answer": 0,
        }
        for language in ("en", "fr", "lt")
        for index in range(10)
    ]
    fixture.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(runner, "EU_MMLU_DATA", fixture)
    monkeypatch.setenv("AISH_MAX_ITEMS_PER_RUN", "5")

    from aish import config

    config.get_settings.cache_clear()
    user_id, target_id = seeded
    run_id = _queue("eu_mmlu", user_id, target_id, sample=False)

    adapter = StubAdapter(lambda _p: "The answer is A.")
    monkeypatch.setattr(runner, "get_adapter", lambda _provider: adapter)
    asyncio.run(runner.execute_run(run_id))

    with session_scope() as db:
        run = db.get(Run, run_id)
        rows_written = list(db.scalars(select(Result).where(Result.run_id == run_id)))

    assert run.status == "failed"
    assert "above this instance's limit" in run.error
    assert rows_written == [], "a refused run must not produce partial evidence"
    assert adapter.calls == [], "a refused run must not call the model"
    config.get_settings.cache_clear()


def test_the_default_limit_admits_a_sampled_eu_mmlu_run():
    """A cap below the smallest real sampled run would make truncation the norm."""
    from aish.config import Settings

    view = current()
    suite = view.suite("eu_mmlu")
    sampled_size = (
        len(suite["languages"])
        * len(suite["dataset"]["expected_subjects_reference"])
        * suite["sampling_mode"]["items_per_stratum"]
    )
    assert Settings().max_items_per_run >= sampled_size
