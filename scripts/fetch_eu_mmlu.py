#!/usr/bin/env python3
"""Provision the EU-MMLU dataset for the harness.

Source: ``EC-DGT-AI/EU-MMLU`` on Hugging Face. The release is one CSV per locale at the
repository root (``eu_mmlu_EN_GB.csv`` ... ``eu_mmlu_SL_SI.csv``), each carrying every split
and subject for that language in the published schema:

    Language, Subject, Split, Index, Question, Choice_0..Choice_3, Answer

This script downloads those files at a pinned commit (or reads local copies), validates every
row, and writes:

    eu_mmlu.jsonl            valid rows, otherwise unmodified (REQ-EUM-01)
    eu_mmlu.rejected.jsonl   rows that failed validation, each with the reason
    eu_mmlu.revision.json    upstream commit, per-file sha256, counts (REQ-EUM-07)

A row is rejected, never repaired, if a field is malformed, or if two rows claim the same
(Language, Subject, Split, Index): the release has such conflicts and there is no way to
know which row is right. The run fails loudly if more than ``--max-rejected-share`` (default 2%) of the
rows are rejected (default 2%): a format change rejects most of a file, while the known
upstream defects are under 1%.

    python3 scripts/fetch_eu_mmlu.py                       # download, validate, write
    python3 scripts/fetch_eu_mmlu.py --from-file DIR|FILE...   # already-downloaded CSVs
    python3 scripts/fetch_eu_mmlu.py --out DIR             # default: $AISH_DATA_DIR or ./data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATASET = "EC-DGT-AI/EU-MMLU"
API = f"https://huggingface.co/api/datasets/{DATASET}"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve"

EXPECTED_COLUMNS = ["Language", "Subject", "Split", "Index", "Question",
                    "Choice_0", "Choice_1", "Choice_2", "Choice_3", "Answer"]
SPLITS = {"dev", "test", "validation"}
CSV_NAME = re.compile(r"^eu_mmlu_([A-Z]{2})_([A-Z]{2})\.csv$")


def read_csv_bytes(raw: bytes, source: str) -> list[dict]:
    """Parse one locale file. The files carry a UTF-8 byte-order mark and questions with
    embedded newlines, so the csv module is used with ``utf-8-sig``."""
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    header = next(reader, [])
    # Some locale files carry empty trailing header cells (",,"). Tolerated only while every
    # data row leaves those cells empty; anything else is a real schema change.
    core = header[: len(EXPECTED_COLUMNS)]
    extra = header[len(EXPECTED_COLUMNS):]
    if core != EXPECTED_COLUMNS or any(name.strip() for name in extra):
        raise SystemExit(
            f"ERROR: {source} columns are {header}, expected {EXPECTED_COLUMNS}. "
            "The published schema has changed; refusing to coerce (REQ-EUM-01)."
        )
    rows = []
    for line, cells in enumerate(reader, start=2):
        if len(cells) == len(header) and any(c.strip() for c in cells[len(EXPECTED_COLUMNS):]):
            row = {name: None for name in EXPECTED_COLUMNS}  # data in an unnamed column
        elif len(cells) >= len(EXPECTED_COLUMNS):
            row = dict(zip(EXPECTED_COLUMNS, cells[: len(EXPECTED_COLUMNS)]))
            if len(cells) > len(header):
                row = {name: None for name in EXPECTED_COLUMNS}
        else:
            row = {name: None for name in EXPECTED_COLUMNS}
        row["_source"], row["_line"] = source, line
        rows.append(row)
    return rows


def validate(rows: list[dict], locale_of: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Split rows into (valid, rejected). Rejected rows carry a ``reason``."""
    accepted, rejected = [], []

    def reject(row: dict, reason: str) -> None:
        rejected.append({**{k: v for k, v in row.items() if not k.startswith("_")},
                         "_source": row["_source"], "_line": row["_line"], "reason": reason})

    candidates = []
    for row in rows:
        if any(v is None for v in row.values()):
            reject(row, "row has too few or too many fields")
        elif not str(row["Index"]).strip().isdigit():
            reject(row, f"Index {row['Index']!r} is not a number")
        elif str(row["Answer"]).strip() not in {"0", "1", "2", "3"}:
            reject(row, f"Answer {row['Answer']!r} is not 0-3")
        elif row["Split"] not in SPLITS:
            reject(row, f"Split {row['Split']!r} is not one of {sorted(SPLITS)}")
        elif not row["Subject"].strip() or not row["Question"].strip() or any(
                not row[f"Choice_{i}"].strip() for i in range(4)):
            reject(row, "empty subject, question or choice")
        elif row["Language"] != locale_of[row["_source"]]:
            reject(row, f"Language {row['Language']!r} does not match its file")
        else:
            candidates.append(row)

    by_key = defaultdict(list)
    for row in candidates:
        by_key[(row["Language"], row["Subject"], row["Split"], int(row["Index"]))].append(row)
    for key, group in by_key.items():
        identical = all(
            {k: v for k, v in g.items() if not k.startswith("_")}
            == {k: v for k, v in group[0].items() if not k.startswith("_")} for g in group)
        if len(group) == 1:
            accepted.append(group[0])
        elif identical:
            accepted.append(group[0])  # an exact repeat carries no ambiguity
        else:
            for row in group:
                reject(row, "conflicting rows share (Language, Subject, Split, Index); "
                            "which is correct cannot be determined")
    return accepted, rejected


def _download(out: Path) -> tuple[list[tuple[str, bytes]], str | None]:
    import httpx

    with httpx.Client(timeout=120.0, follow_redirects=True,
                      headers={"User-Agent": "euiba-aish/0.1"}) as client:
        info = client.get(API)
        info.raise_for_status()
        commit = info.json().get("sha")
        tree = client.get(f"{API}/tree/main")
        tree.raise_for_status()
        names = sorted(e["path"] for e in tree.json() if CSV_NAME.match(e["path"]))
        if not names:
            raise SystemExit("ERROR: no eu_mmlu_<LOCALE>.csv files found; the repository "
                             "layout has changed.")
        files = []
        for name in names:
            response = client.get(f"{RESOLVE}/{commit}/{name}")
            response.raise_for_status()
            files.append((name, response.content))
            print(f"  {name}  {len(response.content):>9,} bytes")
    return files, commit


def _local(paths: list[Path]) -> list[tuple[str, bytes]]:
    files = []
    for path in paths:
        found = sorted(path.glob("eu_mmlu_*.csv")) if path.is_dir() else [path]
        for item in found:
            if not CSV_NAME.match(item.name):
                raise SystemExit(f"ERROR: {item.name} is not an eu_mmlu_<LOCALE>.csv file")
            files.append((item.name, item.read_bytes()))
    if not files:
        raise SystemExit("ERROR: no eu_mmlu_<LOCALE>.csv files found")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from-file", nargs="+", type=Path,
                        help="use already-downloaded CSV file(s) or a folder of them")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("AISH_DATA_DIR", "data")))
    parser.add_argument("--max-rejected-share", type=float, default=0.02)
    args = parser.parse_args()

    if args.from_file:
        files, commit = _local(args.from_file), None
    else:
        print(f"Downloading {DATASET}")
        files, commit = _download(args.out)

    rows, locale_of = [], {}
    for name, raw in files:
        match = CSV_NAME.match(name)
        locale_of[name] = match.group(1)
        rows.extend(read_csv_bytes(raw, name))
    accepted, rejected = validate(rows, locale_of)

    share = len(rejected) / max(1, len(rows))
    if share > args.max_rejected_share:
        print(f"ERROR: {len(rejected)} of {len(rows)} rows rejected ({share:.2%}), above the "
              f"{args.max_rejected_share:.2%} limit. The upstream format has probably changed.",
              file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    clean = [{k: (int(v) if k == "Index" else int(v) if k == "Answer" else v)
              for k, v in row.items() if not k.startswith("_")} for row in accepted]
    with (args.out / "eu_mmlu.jsonl").open("w", encoding="utf-8") as handle:
        for row in clean:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out / "eu_mmlu.rejected.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per = defaultdict(Counter)
    for row in clean:
        per[row["Language"]][row["Split"]] += 1
    (args.out / "eu_mmlu.revision.json").write_text(json.dumps({
        "dataset": DATASET,
        "hub_commit": commit,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files},
        "sha256": hashlib.sha256((args.out / "eu_mmlu.jsonl").read_bytes()).hexdigest(),
        "rows": len(clean),
        "rejected": len(rejected),
        "languages": {lang: dict(c) for lang, c in sorted(per.items())},
        "subjects": dict(Counter(r["Subject"] for r in clean)),
        "licence": "Translations CC BY 4.0 (European Commission DG Translation and EMT Network); "
                   "EN_GB unchanged from MMLU, MIT (Hendrycks et al.)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {len(clean):,} rows to {args.out / 'eu_mmlu.jsonl'}")
    print(f"rejected {len(rejected)} rows -> {args.out / 'eu_mmlu.rejected.jsonl'}")
    for reason, count in Counter(r["reason"].split(" ")[0] + " " + r["reason"].split(" ")[1]
                                 for r in rejected).items():
        print(f"  {count:>3}  {reason}...")
    print(f"languages ({len(per)}): " + ", ".join(
        f"{lang}:{sum(c.values())}" for lang, c in sorted(per.items())))
    print("test-split questions per language: " + ", ".join(
        f"{lang}:{c['test']}" for lang, c in sorted(per.items())))
    print(f"upstream commit: {commit or '(local files; no commit known)'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - network or protocol errors get a clear message
        if exc.__class__.__module__.startswith("httpx"):
            print(f"ERROR: could not reach Hugging Face: {exc}\nDownload the eu_mmlu_*.csv files "
                  "elsewhere and pass them with --from-file.", file=sys.stderr)
            raise SystemExit(1) from None
        raise
