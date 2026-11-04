"""Evidence-checked repairs for the known defects in the EU-MMLU release.

The importer rejects rows it cannot trust. This module then tries to put back, or cleanly
remove, rows whose defect is mechanical and whose correct state can be shown from the rest of
the release. Nothing is repaired on a guess: every rule states its evidence, a fix that fails
its check leaves the row rejected, and every change is returned in a log that is written next
to the data (``eu_mmlu.repairs.jsonl``).

Defects the release is known to contain, and the rule for each:

``stray-index``
    An Index cell holds a number plus stray text (``12 másodperc``). The row is put back at
    that number only if the slot is free and the English row at the same key has the same
    answer and (after normalising decimal commas) at least three identical choices.

``foreign-copy``
    A row's text is identical to another locale's row at the same key and its words belong to
    that other locale's vocabulary (German rows pasted into the Portuguese file, Italian rows
    in the Spanish one). The row is not in its own language, so it is removed. If another row
    holds the slot it stays; otherwise the slot is left empty (the correct text is not
    available) and the removed row goes to the rejected file.

``mislabelled-copy``
    A row is a copy of a same-language row filed under a different Subject at the same Index
    (sociology questions filed as European history), and the English rows at the two keys are
    different questions. Which of the pair is out of place is shown by elimination (the other
    row at the key is not a copy and has the English answer), or, with no partner, by the answer
    (it matches English at the other Subject but not here) or by length (it is the length of the
    English question at the other Subject, not of the one here). Handled like ``foreign-copy``.

``alternate-rendering``
    Two rows share a key, the same answer as English, and are >= 98% the same text (two
    translation drafts of one question). The one with fewer typographic oddities is kept,
    the first in the file on a tie.

Finally, a row in any locale whose answer differs from the English row at the same key is
rejected: the release is a translation, so the answers must agree.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter, defaultdict

CHOICES = [f"Choice_{i}" for i in range(4)]


def text_of(row: dict) -> str:
    return (row["Question"] + " || " + " | ".join(row[c] for c in CHOICES)).strip()


def _key(row: dict) -> tuple:
    return (row["Language"], row["Subject"], row["Split"], int(row["Index"]))


def _public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_") and k != "reason"}


def _norm_choice(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace(",", "."))


def _grams(text: str, n: int = 3) -> Counter:
    text = re.sub(r"\s+", " ", text.lower())
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def _cos(a: Counter, b: Counter) -> float:
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]{3,}", text.lower())


def _oddities(row: dict) -> int:
    t = text_of(row)
    return len(re.findall(r" [.,;:] |\s{2,}|[.,;:]{2,}", t))


def _misplaced_by_length(row: dict, own_ref: dict, other_ref: dict) -> bool:
    """A translation is about as long as its source. A row far longer or shorter than the
    English question at its own key, yet about the length of the English question at the other
    Subject's key, was written for the other key."""
    n = len(text_of(row))
    r_own, r_other = n / max(1, len(text_of(own_ref))), n / max(1, len(text_of(other_ref)))
    return (r_own < 0.4 or r_own > 2.5) and 0.6 <= r_other <= 1.7


def repair(accepted: list[dict], rejected: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (accepted, rejected, repairs). Rows carry Index/Answer as strings or ints;
    the returned accepted rows keep their original types."""
    log: list[dict] = []
    accepted = list(accepted)
    rejected = list(rejected)
    by_key = {_key(r): r for r in accepted}
    en = {k[1:]: r for k, r in by_key.items() if k[0] == "EN"}

    def note(rule: str, action: str, row: dict, evidence: str, **extra) -> None:
        log.append({"rule": rule, "action": action, **_public(row), "evidence": evidence, **extra})

    def reject(row: dict, rule: str, why: str) -> None:
        rejected.append({**{k: v for k, v in row.items() if k != "reason"}, "reason": f"{rule}: {why}"})

    # ---- stray-index ---------------------------------------------------------
    still = []
    for row in rejected:
        m = re.match(r"^\s*(\d+)\s+\S", str(row.get("Index", "")))
        if not (row.get("reason", "").startswith("Index") and m and row["Language"] != "EN"):
            still.append(row)
            continue
        idx = int(m.group(1))
        slot = (row["Language"], row["Subject"], row["Split"], idx)
        ref = en.get((row["Subject"], row["Split"], idx))
        same = (sum(_norm_choice(row[c]) == _norm_choice(ref[c]) for c in CHOICES) if ref else 0)
        if slot not in by_key and ref and str(ref["Answer"]) == str(row["Answer"]) and same >= 3:
            fixed = {**row, "Index": idx}
            fixed.pop("reason", None)
            accepted.append(fixed)
            by_key[slot] = fixed
            note("stray-index", "fixed", row, f"Index {row['Index']!r} -> {idx}: slot free; the English "
                 f"row there has the same answer and {same}/4 identical choices")
        else:
            still.append(row)
    rejected = still

    # ---- vocabularies for language checks -------------------------------------
    # Rows whose text appears verbatim in another locale at the same key are suspects, so
    # they must not contribute to the vocabulary used to judge them.
    seen: dict[tuple, set[str]] = defaultdict(set)
    for r in accepted:
        seen[(r["Subject"], r["Split"], int(r["Index"]), text_of(r))].add(r["Language"])
    vocab: dict[str, Counter] = defaultdict(Counter)
    for r in accepted:
        if len(seen[(r["Subject"], r["Split"], int(r["Index"]), text_of(r))]) == 1:
            vocab[r["Language"]].update(_words(text_of(r)))

    def coverage(row: dict, lang: str) -> float:
        """Share of the row's words found in a language's vocabulary. A row never vouches for
        itself: its own words are subtracted from its own language's counts."""
        words = _words(text_of(row))
        own = Counter(words) if lang == row["Language"] else Counter()
        return sum(vocab[lang][w] - own[w] > 0 for w in words) / max(1, len(words))

    by_text: dict[str, list[dict]] = defaultdict(list)
    for r in accepted:
        if len(text_of(r)) > 60:
            by_text[text_of(r)].append(r)

    def foreign_of(row: dict) -> str | None:
        """The other locale this row's text is a copy of, if its words belong to that locale."""
        if len(text_of(row)) <= 60 or len(_words(text_of(row))) < 8:
            return None
        for other in by_text.get(text_of(row), []):
            if other is row or other["Language"] == row["Language"]:
                continue
            if (other["Subject"], other["Split"], int(other["Index"])) != (
                    row["Subject"], row["Split"], int(row["Index"])):
                continue
            own, theirs = coverage(row, row["Language"]), coverage(row, other["Language"])
            if theirs - own >= 0.15:
                return other["Language"]
        return None

    def copy_of(row: dict, pool: list[dict]) -> dict | None:
        """A same-language row at another Subject, same Split and Index, that is the same text
        (>= 0.9) with the same answer, while the English rows at the two keys are different
        questions. One of the two is in the wrong place; this finds the pair, not which."""
        for other in pool:
            if other is row or other["Language"] != row["Language"] or other["Split"] != row["Split"]:
                continue
            if int(other["Index"]) != int(row["Index"]) or other["Subject"] == row["Subject"]:
                continue
            e1 = en.get((row["Subject"], row["Split"], int(row["Index"])))
            e2 = en.get((other["Subject"], other["Split"], int(other["Index"])))
            if not (e1 and e2 and text_of(e1) != text_of(e2)):
                continue
            if str(other["Answer"]) != str(row["Answer"]):
                continue
            ratio = (1.0 if text_of(other) == text_of(row)
                     else difflib.SequenceMatcher(None, text_of(row), text_of(other)).ratio())
            if ratio >= 0.9:
                return other
        return None

    # ---- conflicting keys ------------------------------------------------------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    rest = []
    for row in rejected:
        if row.get("reason", "").startswith("conflicting"):
            groups[_key(row)].append(row)
        else:
            rest.append(row)
    rejected = rest

    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        survivors, removed = [], []
        pool = accepted + [r for g in groups.values() for r in g]
        ref = en.get(key[1:]) if key[0] != "EN" else None
        for row in rows:
            lang = foreign_of(row)
            if lang:
                removed.append((row, "foreign-copy", f"identical to the {lang} row at this key and "
                                f"its words belong to {lang}, not {row['Language']}"))
            else:
                survivors.append(row)
        # Mislabelled copy, by elimination: of two rows sharing a key, one is a copy of a
        # same-language row filed under another Subject and the other is not, and the other's
        # answer is the English answer at this key. The copy is the one out of place.
        copies = [r for r in survivors if copy_of(r, pool)]
        others = [r for r in survivors if r not in copies]
        if copies and len(others) == 1 and ref and str(ref["Answer"]) == str(others[0]["Answer"]):
            for row in copies:
                other = copy_of(row, pool)
                removed.append((row, "mislabelled-copy", f"copy of the {row['Language']} "
                                f"{other['Subject']} row at Index {row['Index']}; the other row at "
                                "this key is not a copy and has the English answer for this key"))
            survivors = others
        if len(survivors) == 2:
            a, b = survivors
            ratio = difflib.SequenceMatcher(None, text_of(a), text_of(b)).ratio()
            if (ratio >= 0.98 and str(a["Answer"]) == str(b["Answer"])
                    and (not ref or str(ref["Answer"]) == str(a["Answer"]))):
                keep, drop = (b, a) if _oddities(b) < _oddities(a) else (a, b)
                removed.append((drop, "alternate-rendering", f"{ratio:.1%} the same text and the "
                                "same answer as the row kept (fewer typographic oddities, or first)"))
                survivors = [keep]
        if len(survivors) == 1 and (not ref or str(ref["Answer"]) == str(survivors[0]["Answer"])):
            kept = {k: v for k, v in survivors[0].items() if k != "reason"}
            accepted.append(kept)
            by_key[key] = kept
            note("conflict-resolved", "kept", kept, "the one row left at this key after removing "
                 f"{len(removed)} copy/copies; its answer matches the English row")
            for row, rule, why in removed:
                note(rule, "removed", row, why, replaced_by="row kept at the same key")
        else:
            for row in rows:  # could not be resolved: stay rejected, nothing changed
                reject(row, "unresolved", "conflicting rows share a key and no rule could "
                       "establish which is correct")

    # ---- copies with no surviving partner --------------------------------------
    keep = []
    for row in accepted:
        why = None
        lang = foreign_of(row) if row["Language"] != "EN" else None
        if lang:
            why = ("foreign-copy", f"identical to the {lang} row at this key and its words belong to "
                   f"{lang}; the correct {row['Language']} text is not in the release")
        elif row["Language"] != "EN":
            other = copy_of(row, accepted)
            own_ref = en.get((row["Subject"], row["Split"], int(row["Index"])))
            other_ref = en.get((other["Subject"], other["Split"], int(other["Index"]))) if other else None
            # A copy with no partner is removed only when the answer proves which slot it does
            # not belong to: it disagrees with English here and agrees with English at the
            # other Subject. (If the answers happen to coincide, it is left alone.)
            answer_proof = bool(other and own_ref and other_ref
                                and str(row["Answer"]) != str(own_ref["Answer"])
                                and str(row["Answer"]) == str(other_ref["Answer"]))
            length_proof = bool(other and own_ref and other_ref and _misplaced_by_length(
                row, own_ref, other_ref))
            if answer_proof or length_proof:
                proof = ("its answer matches the English %s answer, not the English answer here"
                         % other["Subject"] if answer_proof else
                         "its length is that of the English %s question, not of the English question "
                         "here (a translation keeps roughly the length of its source)" % other["Subject"])
                why = ("mislabelled-copy", f"copy of the {row['Language']} {other['Subject']} row at "
                       f"Index {row['Index']}; {proof}; the correct text for this slot is not in "
                       "the release")
        if why:
            reject(row, *why)
            note(why[0], "quarantined", row, why[1])
        else:
            keep.append(row)
    accepted = keep
    by_key = {_key(r): r for r in accepted}

    # ---- the translation must agree with its source on the answer ---------------
    keep = []
    for row in accepted:
        ref = en.get((row["Subject"], row["Split"], int(row["Index"]))) if row["Language"] != "EN" else None
        if ref and str(ref["Answer"]) != str(row["Answer"]):
            reject(row, "answer-mismatch", "answer differs from the English row at the same key")
            note("answer-mismatch", "quarantined", row, "answer differs from the English row")
        else:
            keep.append(row)
    return keep, rejected, log
