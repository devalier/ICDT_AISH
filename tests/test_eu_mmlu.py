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


def test_sample_is_drawn_from_the_reference_language_and_partial_languages_get_what_they_have(tmp_path, monkeypatch):
    """The release is only partly translated, so a sample cannot require a question to exist in
    every language (that would empty whole subjects). It is drawn from English; each language is
    asked whichever of those questions it has."""
    _write(tmp_path, monkeypatch, drop={("ga", i) for i in range(0, 60, 2)})
    items = runner.build_items(_suite(), current(), sample=True, seed=7)
    idx = {l: {int(i.item_id.rsplit("/", 1)[1]) for i in items if i.variant == l} for l in LANGS}
    assert len(idx["en"]) == 25
    assert idx["de"] == idx["en"]
    assert idx["ga"] < idx["en"] and all(i % 2 == 1 for i in idx["ga"])


def test_only_the_test_split_is_scored(tmp_path, monkeypatch):
    path = tmp_path / "eu_mmlu.jsonl"
    with path.open("w") as handle:
        for split in ("dev", "test", "validation"):
            for i in range(5):
                handle.write(json.dumps({"Language": "en", "Subject": "sociology", "Split": split,
                                         "Index": i, "Question": "q", "Choice_0": "a",
                                         "Choice_1": "b", "Choice_2": "c", "Choice_3": "d",
                                         "Answer": 0}) + "\n")
    monkeypatch.setattr(runner, "EU_MMLU_DATA", path)
    suite = dict(current().suite("eu_mmlu")); suite["languages"] = ["en"]
    assert len(runner.build_items(suite, current(), sample=False, seed=1)) == 5


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


def test_dataset_is_read_from_the_configured_data_directory(tmp_path, monkeypatch):
    """Regression: in the container the app's working directory is read-only /app, and the
    README tells you to copy the dataset into the /data volume. The app must look there."""
    from aish.config import get_settings

    monkeypatch.setenv("AISH_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(runner, "EU_MMLU_DATA", None)
    (tmp_path / "eu_mmlu.jsonl").write_text(
        "\n".join(json.dumps({
            "Language": lang, "Subject": "sociology", "Split": "test", "Index": i,
            "Question": "q", "Choice_0": "a", "Choice_1": "b", "Choice_2": "c",
            "Choice_3": "d", "Answer": 0}) for lang in LANGS for i in range(3)),
        encoding="utf-8")
    items = runner.build_items(_suite(), current(), sample=False, seed=1)
    assert len(items) == 9


def test_missing_dataset_message_names_the_expected_path(tmp_path, monkeypatch):
    from aish.config import get_settings

    monkeypatch.setenv("AISH_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr("aish.config.BASE_DIR", tmp_path / "nowhere")  # no bundled snapshot
    monkeypatch.setattr(runner, "EU_MMLU_DATA", None)
    with pytest.raises(RunError) as err:
        runner.build_items(_suite(), current(), sample=False, seed=1)
    assert str(tmp_path / "eu_mmlu.jsonl") in str(err.value)
    assert "--from-file" in str(err.value)


def test_readiness_follows_the_data_directory(client, register, tmp_path, monkeypatch):
    from aish.config import get_settings

    monkeypatch.setenv("AISH_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr("aish.config.BASE_DIR", tmp_path / "nowhere")  # no bundled snapshot
    register(client)
    page = client.get("/suites/eu_mmlu").text
    assert "Dataset provisioned on this server" in page and "missing" in page
    assert "Dataset revision pinned in the suite file" in page
    (tmp_path / "eu_mmlu.jsonl").write_text("{}\n")
    (tmp_path / "eu_mmlu.revision.json").write_text(json.dumps({"rows": 17200, "sha256": "ab" * 32}))
    page = client.get("/suites/eu_mmlu").text
    assert "17,200 rows" in page
