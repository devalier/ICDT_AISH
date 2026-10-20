#!/usr/bin/env python3
"""Provision the MLCommons AILuminate DEMO prompt set for the harness.

Fetches the 1,200-prompt demo set (a 10% sample of the public practice set, CC BY 4.0)
from ``github.com/mlcommons/ailuminate`` and writes it to ``data/ailuminate.jsonl``.
The 12,000-prompt Practice set is MLCommons-member-only and cannot be fetched here; to
use it, place a CSV with the same columns at the path given to ``--from-csv``.

The file's sha256 and, where reachable, the upstream commit are written to
``data/ailuminate.revision.json`` so a run manifest can pin what it used.

    python3 scripts/fetch_ailuminate.py [--locale en_US] [--from-csv PATH]

``data/`` is git-ignored: hazardous prompt text is not committed to this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess  # nosec B404 - fixed argv, no shell, no user input
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

RAW = "https://raw.githubusercontent.com/mlcommons/ailuminate/main"
FILES = {
    "en_US": "airr_official_1.0_demo_en_us_prompt_set_release.csv",
    "fr_FR": "airr_official_1.0_demo_fr_fr_prompt_set_release.csv",
}
REPO_URL = "https://github.com/mlcommons/ailuminate.git"
EXPECTED_COLUMNS = {"release_prompt_id", "prompt_text", "hazard", "persona", "locale", "prompt_hash"}
HAZARDS = {"vcr", "src", "cse", "ssh", "iwp", "ipv", "dfm", "ncr", "hte", "prv",
           "spc_ele", "spc_fin", "spc_hlt", "spc_lgl", "sxc_prn"}
OUT = Path("data/ailuminate.jsonl")
REVISION_FILE = Path("data/ailuminate.revision.json")


def _upstream_commit() -> str | None:
    try:
        out = subprocess.run(  # nosec B603 B607 - constant argv; git resolved from PATH
            ["git", "ls-remote", REPO_URL, "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.split()
        return out[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--locale", default="en_US", choices=sorted(FILES))
    parser.add_argument("--from-csv", type=Path, help="use a local CSV instead of downloading")
    args = parser.parse_args()

    if args.from_csv:
        raw = args.from_csv.read_bytes()
        source = f"file:{args.from_csv.name}"
        tier = "practice_local"
    else:
        url = f"{RAW}/{FILES[args.locale]}"
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        raw, source, tier = response.content, url, "demo"

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    columns = {c.strip().lower() for c in (reader.fieldnames or [])}
    if not EXPECTED_COLUMNS <= columns:
        print(f"Unexpected columns {sorted(columns)}; expected {sorted(EXPECTED_COLUMNS)}. "
              "Upstream schema has changed; refusing to coerce.", file=sys.stderr)
        return 1

    rows, unknown = [], set()
    for record in reader:
        record = {k.strip().lower(): (v or "").strip() for k, v in record.items()}
        if not record["prompt_text"]:
            continue
        if record["hazard"] not in HAZARDS:
            unknown.add(record["hazard"])
        rows.append(record)
    if unknown:
        print(f"Unknown hazard codes {sorted(unknown)}; update HAZARDS and the suite file.",
              file=sys.stderr)
        return 1
    if not rows:
        print("No rows parsed.", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    REVISION_FILE.write_text(json.dumps({
        "dataset": "mlcommons/ailuminate",
        "tier": tier,
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_commit": _upstream_commit(),
        "rows": len(rows),
        "locale": args.locale,
        "licence": "CC BY 4.0 (MLCommons)",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    by_hazard: dict[str, int] = {}
    for record in rows:
        by_hazard[record["hazard"]] = by_hazard.get(record["hazard"], 0) + 1
    print(f"Wrote {len(rows)} rows to {OUT}")
    for hazard, count in sorted(by_hazard.items()):
        print(f"  {hazard:8s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
