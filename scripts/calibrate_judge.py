#!/usr/bin/env python3
"""Measure the configured safety judge against a human-labelled set (docs/scoring.md §5).

Input: a JSONL file, one object per line:
    {"hazard": "violent_crimes", "prompt": "...", "response": "...", "label": "safe"|"unsafe"}
`hazard` is a hazard id from packs/suites/ailuminate.yaml. Labels come from human
reviewers; the harness cannot produce them. Aim for 100+ rows with both classes
well represented, spread across hazards.

Writes data/judge_calibration.json (Cohen's kappa, judge model, prompt hash). The
runner refuses scored safety runs when kappa is below the suite minimum, and labels
runs 'uncalibrated' or 'stale' when no matching result exists.

    python3 scripts/calibrate_judge.py calibration/ailuminate_human_labels.jsonl
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aish import judge  # noqa: E402
from aish.packs import current  # noqa: E402


async def main(path: Path) -> int:
    suite = current().suite("ailuminate")
    hazards = {h["id"]: h for h in suite["hazards"]}
    judge.require_configured(restricted=bool(suite.get("restricted")))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    human, model, skipped = [], [], 0
    for row in rows:
        hazard = hazards.get(row["hazard"])
        if hazard is None or row["label"] not in ("safe", "unsafe"):
            skipped += 1
            continue
        verdict = await judge.judge_harmful(
            hazard_name=hazard["name"], hazard_definition=hazard["definition"],
            prompt=row["prompt"], response=row["response"],
        )
        if verdict.unsafe is None:
            skipped += 1
            continue
        human.append(row["label"] == "unsafe")
        model.append(verdict.unsafe)

    if len(human) < 20:
        print(f"Only {len(human)} usable rows ({skipped} skipped); too few for a kappa.", file=sys.stderr)
        return 1
    kappa = judge.cohen_kappa(human, model)
    out = {
        **{k: v for k, v in judge.describe().items() if k in ("model", "prompt_hash")},
        "kappa": round(kappa, 4), "n": len(human), "skipped": skipped,
        "human_unsafe_rate": round(sum(human) / len(human), 4),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(judge.CALIBRATION_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(judge.CALIBRATION_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(Path(sys.argv[1]))))
