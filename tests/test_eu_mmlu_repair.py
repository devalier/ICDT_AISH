"""The EU-MMLU repair rules, on synthetic data, one rule at a time."""

from __future__ import annotations

import sys
from itertools import count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eu_mmlu_repair as repair_mod  # noqa: E402
import fetch_eu_mmlu as importer  # noqa: E402

LETTERS = "abcdefghij"


def word(lang: str, n: int) -> str:
    """A pseudo-word that belongs to one language's vocabulary (letters only)."""
    return lang.lower() + "x" + "".join(LETTERS[int(d)] for d in str(n))


def sentence(lang: str, seed: int, length: int = 30) -> str:
    # Drawn from a small per-language pool, so a language's words recur across its rows the way
    # real vocabulary does, while different seeds still give different sentences.
    return " ".join(word(lang, (seed * 7 + i * 13) % 80) for i in range(length))


_line = count(2)


def mk(lang, subject, index, seed, answer=1, source=None, text_lang=None, choices=None):
    tl = text_lang or lang
    row = {"Language": lang, "Subject": subject, "Split": "test", "Index": str(index),
           "Question": sentence(tl, seed),
           "Answer": str(answer), "_source": source or f"eu_mmlu_{lang}_{lang}.csv", "_line": next(_line)}
    ch = choices or [word(tl, 200 + seed * 7 + k) for k in range(4)]
    for k in range(4):
        row[f"Choice_{k}"] = ch[k]
    return row


def corpus():
    """Enough rows per language for vocabularies, and English as the answer source."""
    rows = []
    for lang in ("EN", "DE", "PT", "ES", "IT"):
        for i in range(30):
            rows.append(mk(lang, "sociology", i, seed=i, answer=i % 4))
        for i in range(30):
            rows.append(mk(lang, "history", i, seed=100 + i, answer=(i + 1) % 4))
    return rows


def run(rows):
    locale_of = {r["_source"]: r["Language"] for r in rows}
    accepted, rejected = importer.validate(rows, locale_of)
    return importer.repair(accepted, rejected)


def find(rows, lang, subject, index):
    return [r for r in rows if r["Language"] == lang and r["Subject"] == subject and int(r["Index"]) == index]


def test_a_clean_corpus_is_left_alone():
    accepted, rejected, repairs = run(corpus())
    assert len(accepted) == len(corpus()) and not rejected and not repairs


def test_stray_text_in_index_is_fixed_only_with_evidence_from_the_english_row():
    rows = corpus()
    rows.remove(find(rows, "PT", "sociology", 12)[0])
    en = find(rows, "EN", "sociology", 12)[0]
    bad = mk("PT", "sociology", "12 segundos", seed=12, answer=en["Answer"],
             choices=[c for c in (en[f"Choice_{k}"] for k in range(4))])
    accepted, rejected, repairs = run(rows + [bad])
    fixed = find(accepted, "PT", "sociology", 12)
    assert len(fixed) == 1 and [r["rule"] for r in repairs] == ["stray-index"]
    assert not rejected

    # Same row, but the English answer disagrees: nothing is guessed, it stays rejected.
    bad2 = dict(bad, Answer=str((int(en["Answer"]) + 1) % 4))
    accepted, rejected, repairs = run(rows + [bad2])
    assert not find(accepted, "PT", "sociology", 12) and len(rejected) == 1 and not repairs


def test_foreign_copy_with_a_partner_is_removed_and_the_native_row_kept():
    rows = corpus()
    native = find(rows, "PT", "sociology", 5)[0]
    german = dict(find(rows, "DE", "sociology", 5)[0], Language="PT", _source=native["_source"])
    accepted, rejected, repairs = run(rows + [german])
    kept = find(accepted, "PT", "sociology", 5)
    assert len(kept) == 1 and kept[0]["Question"] == native["Question"]
    assert not rejected
    assert {(r["rule"], r["action"]) for r in repairs} == {("conflict-resolved", "kept"), ("foreign-copy", "removed")}


def test_foreign_copy_without_a_partner_is_quarantined_not_invented():
    rows = corpus()
    rows.remove(find(rows, "ES", "sociology", 7)[0])
    italian = dict(find(rows, "IT", "sociology", 7)[0], Language="ES", _source="eu_mmlu_ES_ES.csv")
    accepted, rejected, repairs = run(rows + [italian])
    assert not find(accepted, "ES", "sociology", 7)
    assert len(rejected) == 1 and rejected[0]["reason"].startswith("foreign-copy")
    assert repairs[0]["action"] == "quarantined"


def test_a_mislabelled_copy_is_removed_by_elimination():
    """History row is a copy of the sociology row at the same Index; the other row at the key
    is not a copy and carries the English history answer."""
    rows = corpus()
    soc = find(rows, "ES", "sociology", 9)[0]
    copy = dict(soc, Subject="history")
    true_hist = find(rows, "ES", "history", 9)[0]
    rows.remove(true_hist)
    accepted, rejected, repairs = run(rows + [copy, true_hist])
    hist = find(accepted, "ES", "history", 9)
    assert len(hist) == 1 and hist[0]["Question"] == true_hist["Question"]
    assert not rejected and {r["rule"] for r in repairs} == {"conflict-resolved", "mislabelled-copy"}


def test_a_mislabelled_copy_with_no_partner_needs_the_answer_to_prove_it():
    rows = corpus()
    soc = find(rows, "ES", "sociology", 4)[0]
    en_hist_answer = find(rows, "EN", "history", 4)[0]["Answer"]
    rows.remove(find(rows, "ES", "history", 4)[0])
    proven = dict(soc, Subject="history")       # answer is the sociology answer, not the history one
    assert str(soc["Answer"]) != str(en_hist_answer)
    accepted, rejected, _ = run(rows + [proven])
    assert not find(accepted, "ES", "history", 4) and len(rejected) == 1


def test_alternate_renderings_keep_the_cleaner_row():
    rows = corpus()
    row = find(rows, "ES", "sociology", 3)[0]
    messy = dict(row, Question=row["Question"].replace(" ", " . ", 1), _line=next(_line))
    accepted, rejected, repairs = run(rows + [messy])
    kept = find(accepted, "ES", "sociology", 3)
    assert len(kept) == 1 and kept[0]["Question"] == row["Question"]
    assert any(r["rule"] == "alternate-rendering" for r in repairs) and not rejected


def test_unprovable_conflicts_stay_rejected_and_nothing_is_changed():
    rows = corpus()
    other = mk("ES", "sociology", 2, seed=999, answer=0)          # a different question, same key
    accepted, rejected, repairs = run(rows + [other])
    assert not find(accepted, "ES", "sociology", 2)
    assert len(rejected) == 2 and all(r["reason"].startswith("unresolved") for r in rejected)
    assert not repairs


def test_an_answer_that_disagrees_with_english_is_rejected():
    rows = corpus()
    row = find(rows, "ES", "sociology", 6)[0]
    row["Answer"] = str((int(find(rows, "EN", "sociology", 6)[0]["Answer"]) + 1) % 4)
    accepted, rejected, _ = run(rows)
    assert not find(accepted, "ES", "sociology", 6)
    assert rejected[0]["reason"].startswith("answer-mismatch")
