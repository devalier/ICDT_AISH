"""The EU-MMLU importer and the bundled snapshot, checked against the real release."""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from aish import runner
from aish.packs import current
from test_runner import StubAdapter, _execute, _queue, seeded  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_eu_mmlu as importer  # noqa: E402

SNAPSHOT = ROOT / "datasets" / "eu_mmlu"
COLS = importer.EXPECTED_COLUMNS


def _csv(rows, header=None) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(header or COLS)
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")  # the release carries a BOM


def _row(lang="EN", subject="sociology", split="test", index=0, q="Question?", answer=1):
    return [lang, subject, split, index, q, "a", "b", "c", "d", answer]


# ---- importer ------------------------------------------------------------- #


def test_parses_bom_embedded_newlines_and_empty_trailing_header_cells():
    raw = _csv([_row(q="line one\nline two, with comma") + ["", ""]], header=COLS + ["", ""])
    rows = importer.read_csv_bytes(raw, "eu_mmlu_EN_GB.csv")
    assert len(rows) == 1 and rows[0]["Question"] == "line one\nline two, with comma"
    accepted, rejected = importer.validate(rows, {"eu_mmlu_EN_GB.csv": "EN"})
    assert len(accepted) == 1 and not rejected


def test_schema_change_fails_loudly():
    with pytest.raises(SystemExit):
        importer.read_csv_bytes(_csv([], header=COLS[:-1] + ["Correct"]), "x.csv")
    with pytest.raises(SystemExit):
        importer.read_csv_bytes(_csv([], header=COLS + ["Notes"]), "x.csv")


def test_malformed_rows_are_rejected_with_a_reason_not_repaired():
    rows = importer.read_csv_bytes(_csv([
        _row(index="12 másodperc"), _row(index=1, answer=7), _row(index=2, split="train"),
        _row(index=3, lang="DE"), _row(index=4), _row(index=5, q=""),
    ]), "eu_mmlu_EN_GB.csv")
    accepted, rejected = importer.validate(rows, {"eu_mmlu_EN_GB.csv": "EN"})
    assert [int(r["Index"]) for r in accepted] == [4]
    reasons = " | ".join(r["reason"] for r in rejected)
    for expected in ("not a number", "not 0-3", "Split", "does not match its file", "empty"):
        assert expected in reasons


def test_conflicting_keys_are_both_rejected_but_exact_repeats_kept_once():
    rows = importer.read_csv_bytes(_csv([
        _row(index=1, q="Which?"), _row(index=1, q="A different question"),   # conflict
        _row(index=2, q="Same"), _row(index=2, q="Same"),                      # exact repeat
    ]), "eu_mmlu_EN_GB.csv")
    accepted, rejected = importer.validate(rows, {"eu_mmlu_EN_GB.csv": "EN"})
    assert [int(r["Index"]) for r in accepted] == [2]
    assert len(rejected) == 2 and all("conflicting" in r["reason"] for r in rejected)


# ---- the bundled snapshot -------------------------------------------------- #


@pytest.fixture(scope="module")
def snapshot():
    # Iterate the file, not str.splitlines(): question text can contain U+2028/U+0085.
    with (SNAPSHOT / "eu_mmlu.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_snapshot_matches_the_pin_and_the_published_schema(snapshot):
    rev = json.loads((SNAPSHOT / "eu_mmlu.revision.json").read_text())
    assert rev["hub_commit"] == current().suite("eu_mmlu")["dataset"]["revision"]
    assert rev["rows"] == len(snapshot) == 17120 and rev["rejected"] == 15
    assert all(list(r) == COLS for r in snapshot[:50])
    assert all(isinstance(r["Index"], int) and r["Answer"] in (0, 1, 2, 3) for r in snapshot)
    keys = Counter((r["Language"], r["Subject"], r["Split"], r["Index"]) for r in snapshot)
    assert max(keys.values()) == 1, "snapshot must contain no ambiguous keys"


def test_snapshot_covers_the_sixteen_locales_and_eight_subjects(snapshot):
    suite = current().suite("eu_mmlu")
    wanted = {l.upper() for l in suite["languages"]}
    test_rows = [r for r in snapshot if r["Split"] == "test"]
    assert wanted <= {r["Language"] for r in test_rows}
    assert "ES" not in wanted and len(wanted) == 16
    assert len({r["Subject"] for r in snapshot}) == 8
    by = Counter(r["Language"] for r in test_rows)
    assert by["EN"] == 1036 and by["GA"] == 391          # the release is partly translated
    assert by["HU"] == 618 and by["PT"] == 776


def test_index_aligns_across_languages_so_paired_parity_is_valid(snapshot):
    """Same (Subject, Split, Index) is the same question: every language's answer, Spanish
    included now that its defects are repaired, equals the English one. This is the assumption
    paired parity rests on."""
    en = {(r["Subject"], r["Split"], r["Index"]): r["Answer"] for r in snapshot if r["Language"] == "EN"}
    wanted = {r["Language"] for r in snapshot} - {"EN"}
    assert len(wanted) == 16
    checked = 0
    for r in snapshot:
        if r["Language"] in wanted:
            key = (r["Subject"], r["Split"], r["Index"])
            assert key in en and en[key] == r["Answer"], (r["Language"], key)
            checked += 1
    assert checked > 10000


# ---- end to end on the real data ------------------------------------------- #


def test_a_perfect_model_scores_full_marks_with_paired_parity_on_the_real_data(seeded, monkeypatch):  # noqa: F811
    suite = current().suite("eu_mmlu")
    items = runner.build_items(suite, current(), sample=True, seed=20260921)
    assert 2500 < len(items) < 5000
    answers = {i.prompt: "ABCD"[i.meta["answer"]] for i in items}
    assert len(answers) == len(items)                      # prompts are unique per question

    user_id, target_id = seeded
    run, rows, summary = _execute(_queue("eu_mmlu", user_id, target_id, sample=True),
                                  StubAdapter(lambda prompt: answers[prompt]), monkeypatch)
    assert run.status == "done", run.error
    assert summary["metrics"]["accuracy"] == 1.0 and summary["metrics"]["unparsed_rate"] == 0.0
    assert set(summary["parity"]) == {l for l in suite["languages"] if l != "en"}
    assert all(v == pytest.approx(1.0) for v in summary["parity"].values())
    assert set(summary["parity_basis"].values()) == {"paired"}
    # Irish only has four of the eight subjects; it is paired on what it shares with English.
    assert summary["parity_n"]["ga"] < summary["parity_n"]["de"]
    assert summary["dataset_revision"] if "dataset_revision" in summary else True


def test_the_repaired_defects_are_fixed_and_the_rest_are_documented(snapshot):
    by = {(r["Language"], r["Subject"], r["Split"], r["Index"]): r for r in snapshot}
    # Hungarian: the row with "12 másodperc" in its Index is back at Index 12, and matches English.
    hu = by[("HU", "college_chemistry", "test", 12)]
    assert "ciklobutil" in hu["Question"] and hu["Answer"] == by[("EN", "college_chemistry", "test", 12)]["Answer"]
    # Portuguese: the stray German rows are gone and the subject is as complete as English.
    sociology = [r for r in snapshot if r["Language"] == "PT" and r["Subject"] == "sociology" and r["Split"] == "test"]
    assert len(sociology) == 201 and not any(r["Language"] == "PT" and "Laut Foucault" in r["Question"] for r in snapshot)
    # Spanish: no Italian text, and no row is a copy of another subject's row.
    assert not any(r["Language"] == "ES" and r["Question"].startswith("Questa domanda") for r in snapshot)
    soc = {(r["Split"], r["Index"]): r["Question"] for r in snapshot
           if r["Language"] == "ES" and r["Subject"] == "sociology"}
    copies = [r for r in snapshot if r["Language"] == "ES" and r["Subject"] == "high_school_european_history"
              and soc.get((r["Split"], r["Index"])) == r["Question"]]
    assert not copies
    # What could not be repaired is quarantined with a reason, and everything changed is logged.
    with (SNAPSHOT / "eu_mmlu.rejected.jsonl").open(encoding="utf-8") as handle:
        rejected = [json.loads(line) for line in handle if line.strip()]
    assert len(rejected) == 15 and {r["Language"] for r in rejected} == {"ES"}
    assert all(r["reason"].split(":")[0] in ("foreign-copy", "mislabelled-copy") for r in rejected)
    with (SNAPSHOT / "eu_mmlu.repairs.jsonl").open(encoding="utf-8") as handle:
        repairs = [json.loads(line) for line in handle if line.strip()]
    assert len(repairs) == 146 and all(r["evidence"] for r in repairs)
