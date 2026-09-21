#!/usr/bin/env python3
"""Provision the EU-MMLU dataset for the harness.

Downloads ``EC-DGT-AI/EU-MMLU`` through the Hugging Face datasets server and
writes it to ``data/eu_mmlu.jsonl`` in the published schema, unmodified
(REQ-EUM-01). The resolved dataset revision is written alongside it so a run
manifest can pin what it used (REQ-EUM-07).

    python3 scripts/fetch_eu_mmlu.py [--limit N] [--split test]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

DATASET = "EC-DGT-AI/EU-MMLU"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
INFO_URL = "https://datasets-server.huggingface.co/info"
SPLITS_URL = "https://datasets-server.huggingface.co/splits"
PAGE = 100  # the datasets server caps a single page at 100 rows

EXPECTED_COLUMNS = {
    "Language", "Subject", "Split", "Index", "Question",
    "Choice_0", "Choice_1", "Choice_2", "Choice_3", "Answer",
}

OUT = Path("data/eu_mmlu.jsonl")
REVISION_FILE = Path("data/eu_mmlu.revision.json")


def _get(client: httpx.Client, url: str, params: dict) -> dict:
    response = client.get(url, params=params, timeout=60.0)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 fetches every row")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True, headers={"User-Agent": "euiba-aish/0.1"}) as client:
        config = args.config
        splits = _get(client, SPLITS_URL, {"dataset": DATASET})
        available = splits.get("splits", [])
        if config is None:
            config = available[0]["config"] if available else "default"
        split_names = sorted({s["split"] for s in available if s["config"] == config})
        split = args.split if args.split in split_names else (split_names[0] if split_names else args.split)
        print(f"dataset={DATASET} config={config} split={split}")

        info = _get(client, INFO_URL, {"dataset": DATASET, "config": config})
        dataset_info = info.get("dataset_info") or {}
        columns = set((dataset_info.get("features") or {}).keys())
        if columns and not EXPECTED_COLUMNS.issubset(columns):
            # Fail loudly on a schema change rather than coercing columns (REQ-EUM-01).
            missing = sorted(EXPECTED_COLUMNS - columns)
            print(f"ERROR: published schema changed; missing columns: {missing}", file=sys.stderr)
            return 2

        first = _get(client, ROWS_URL, {"dataset": DATASET, "config": config,
                                        "split": split, "offset": 0, "length": PAGE})
        total = int(first.get("num_rows_total") or 0)
        target = min(total, args.limit) if args.limit else total
        print(f"rows available={total} fetching={target}")

        written = 0
        with OUT.open("w", encoding="utf-8") as handle:
            page = first
            offset = 0
            while written < target:
                rows = page.get("rows") or []
                if not rows:
                    break
                for entry in rows:
                    if written >= target:
                        break
                    handle.write(json.dumps(entry["row"], ensure_ascii=False) + "\n")
                    written += 1
                offset += len(rows)
                if written >= target:
                    break
                page = _get(client, ROWS_URL, {"dataset": DATASET, "config": config,
                                               "split": split, "offset": offset, "length": PAGE})
                if offset % 2000 == 0:
                    print(f"  {written}/{target}")

    REVISION_FILE.write_text(
        json.dumps(
            {"dataset": DATASET, "config": config, "split": split, "rows": written,
             "source": "huggingface datasets-server"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {written} rows to {OUT}")
    print(f"Pin this in packs/suites/eu_mmlu.yaml -> dataset.revision before a scored run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.HTTPError as exc:
        print(
            f"ERROR: could not reach the Hugging Face datasets server: {exc}\n"
            "If this host is behind an egress policy, download the dataset on a "
            "machine that can reach huggingface.co and copy data/eu_mmlu.jsonl across.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
