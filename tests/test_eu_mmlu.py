"""EU-MMLU: answer encoding, language-matched sampling, paired parity."""

from __future__ import annotations

import json

import pytest

from aish import runner
from aish.packs import current
from aish.runner_types import RunError
from test_runner import StubAdapter, _execute, _queue, seeded  # noqa: F401

LANGS = ["en", "de", "ga"]


def _write(tmp_path, monkeypatch, *, answer=lambda i: i % 4, drop=()):
    path = tmp_path / "eu_mmlu.jsonl"
    with path.open("w") as handle:
        for lang in LANGS:
            for i in range(60):
                if (lang, i) in drop:
                    continue
                handle.write(json.dumps({
                    "Language": lang, "Subject": "sociology", "Split": "test", "Index": i,
                    "Question": f"Q{i} in {lang}", "Choice_0": "a0", "Choice_1": "b1",
                    "Choice_2": "c2", "Choice_3": "d3", "Answer": answer(i),
                }) + "\n")
    monkeypatch.setattr(runner, "EU_MMLU_DATA", path)


def _suite():
    suite = dict(current().suite("eu_mmlu"))
    suite["languages"] = LANGS
    return suite


@pytest.mark.parametrize("encode", [lambda i: i % 4, lambda i: str(i % 4), lambda i: "ABCD"[i % 4]])
def test_all_plausible_answer_encodings_are_normalised(tmp_path, monkeypatch, encode):
    _write(tmp_path, monkeypatch, answer=encode)
    items = runner.build_items(_suite(), current(), sample=False, seed=1)
    assert all(i.meta["answer"] == int(i.item_id.rsplit("/", 1)[1]) % 4 for i in items)


def test_unrecognised_answer_encoding_fails_loudly(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, answer=lambda i: "the second one")
    with pytest.raises(RunError, match="Answer="):
        runner.build_items(_suite(), current(), sample=False, seed=1)


def test_sample_mode_picks_the_same_questions_in_every_language(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    items = runner.build_items(_suite(), current(), sample=True, seed=7)
    by_lang = {l: {i.item_id.rsplit("/", 1)[1] for i in items if i.variant == l} for l in LANGS}
    assert len(by_lang["en"]) == 25
    assert by_lang["en"] == by_lang["de"] == by_lang["ga"]
    # Reproducible.
    again = runner.build_items(_suite(), current(), sample=True, seed=7)
    assert [i.item_id for i in again] == [i.item_id for i in items]


def test_sample_is_restricted_to_questions_present_in_all_languages(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, drop={("ga", i) for i in range(0, 60, 2)})
    items = runner.build_items(_suite(), current(), sample=True, seed=7)
    assert all(int(i.item_id.rsplit("/", 1)[1]) % 2 == 1 for i in items)


def test_parity_is_computed_on_the_paired_item_set(seeded, tmp_path, monkeypatch):  # noqa: F811
    """A model right on every question in English but wrong on all even-indexed ones in
    German: paired parity must be exactly 0.5, even though the German set differs in size."""
    _write(tmp_path, monkeypatch, drop={("de", i) for i in range(50, 60)})
    user_id, target_id = seeded

    def reply(prompt: str) -> str:
        # Choices are fixed a0..d3 and the answer is Index % 4; recover the index from Q.
        index = int(prompt.split("Q", 1)[1].split(" ", 1)[0])
        letter = "ABCD"[index % 4]
        if " in de" in prompt and index % 2 == 0:
            letter = "ABCD"[(index + 1) % 4]
        return letter

    run, rows, summary = _execute(_queue("eu_mmlu", user_id, target_id, sample=False),
                                  StubAdapter(reply), monkeypatch)
    assert run.status == "done", run.error
    assert summary["parity_basis"]["de"] == "paired"
    assert summary["parity"]["de"] == pytest.approx(0.5)
    assert summary["parity"]["ga"] == pytest.approx(1.0)
    assert summary["metrics"]["parity_floor"] == pytest.approx(0.5)
