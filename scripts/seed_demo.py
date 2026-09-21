#!/usr/bin/env python3
"""Populate a development database with a demo account and finished runs.

Refuses to run in production. It exists so the panes can be reviewed without
spending money on real model calls, and it never touches a real provider.

    AISH_DATABASE_URL=sqlite:///data/dev.db python3 scripts/seed_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aish import runner  # noqa: E402
from aish.config import get_settings  # noqa: E402
from aish.db import init_db, session_scope  # noqa: E402
from aish.models import ModelTarget, Run, User  # noqa: E402
from aish.providers.base import Completion  # noqa: E402
from aish.security import encrypt_secret, hash_password  # noqa: E402

DEMO_EMAIL = "demo@ec.europa.eu"
# Development fixture only; the script refuses to run in production.
DEMO_PASSWORD = "demo-passphrase-for-review"  # nosec B105


class ScriptedModel:
    """A stand-in model with deliberately uneven behaviour, so each pane has
    something to show."""

    def __init__(self, name: str, weak_language: str, declined_nationality: str, honest: bool):
        self.name = name
        self.weak_language = weak_language
        self.declined_nationality = declined_nationality
        self.honest = honest

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout):
        return Completion(text=self._reply(prompt), tokens_in=120, tokens_out=40, latency_ms=380)

    def _reply(self, prompt: str) -> str:
        if "Answer with a single letter" in prompt:
            return "The answer is B." if f"in {self.weak_language}?" in prompt else "The answer is A."
        if "recruitment panel" in prompt:
            return ("DO NOT RECOMMEND. The profile is difficult to verify."
                    if f"a {self.declined_nationality} national" in prompt
                    else "RECOMMEND. Strong policy background.")
        if "hardship grant" in prompt:
            return "REFER" if "roma" in prompt else "APPROVE"
        if "2027/1180" in prompt or "C-994/24" in prompt:
            return ("I cannot find any record of that instrument; it does not appear to exist."
                    if self.honest else
                    "It imposes three core obligations on municipal registries, chiefly "
                    "quarterly reporting and a data-quality audit.")
        if "right of legislative initiative" in prompt:
            return "The European Commission holds the right of initiative."
        if "state the figure reported for 2024" in prompt:
            return ("The supplied text does not contain a 2024 figure."
                    if self.honest else "The figure reported for 2024 is 5,702.")
        if "European Ombudsman" in prompt:
            return ("Submit the complaint online within two years of becoming aware of the "
                    "facts. You will need your contact details, the body concerned and a "
                    "description of the decision. Expect an acknowledgement within a week.")
        if "performance summary" in prompt:
            return ("A capable and analytical engineer who shipped two major systems and "
                    "mentors juniors. Occasionally misses internal deadlines.")
        return "A short answer describing the main festivals and their significance."


def fixture_eu_mmlu() -> Path:
    """A tiny local stand-in so the parity pane has data without the 17k-row
    download. Clearly not the real dataset."""
    path = Path("data/eu_mmlu.demo.jsonl")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for language in ["en", "fr", "de", "pl", "lt", "el", "ga"]:
        for index in range(6):
            rows.append(
                {
                    "Language": language, "Subject": "international_law", "Split": "test",
                    "Index": index,
                    "Question": f"Demo question {index} in {language}?",
                    "Choice_0": "only armed force",
                    "Choice_1": "all forms of force including sanctions",
                    "Choice_2": "any interference in domestic affairs",
                    "Choice_3": "only force undermining territorial integrity",
                    "Answer": 0,
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def main() -> int:
    settings = get_settings()
    if settings.is_production:
        print("refusing to seed demo data in production", file=sys.stderr)
        return 2

    init_db()
    runner.EU_MMLU_DATA = fixture_eu_mmlu()

    scripted = {
        "Llama 3.3 70B (on-prem)": ScriptedModel("llama", "lt", "RO", honest=True),
        "GPT 5.1 (cloud)": ScriptedModel("gpt", "ga", "BG", honest=False),
    }

    with session_scope() as db:
        user = db.query(User).filter(User.email == DEMO_EMAIL).one_or_none()
        if user is None:
            user = User(
                email=DEMO_EMAIL,
                password_hash=hash_password(DEMO_PASSWORD),
                display_name="Demo Reviewer",
                euiba_id="ICDT",
                euiba_name="Interinstitutional Committee for Digital Transformation",
                is_admin=True,
            )
            db.add(user)
            db.flush()
        user_id = user.id

        target_ids = {}
        for label in scripted:
            target = db.query(ModelTarget).filter(
                ModelTarget.user_id == user_id, ModelTarget.label == label
            ).one_or_none()
            if target is None:
                target = ModelTarget(
                    user_id=user_id,
                    label=label,
                    provider="openai_compatible",
                    model_name=label.split(" (")[0].lower().replace(" ", "-"),
                    endpoint="https://example.com/v1",
                    api_key_ciphertext=encrypt_secret("sk-demo-key-0000", f"user:{user_id}"),
                    api_key_hint="****0000",
                )
                db.add(target)
                db.flush()
            target_ids[label] = target.id

    from aish.packs import current

    view = current()
    for label, model in scripted.items():
        for suite_id in ("eu_mmlu", "euiba_bias", "euiba_grounding"):
            suite = view.suite(suite_id)
            with session_scope() as db:
                existing = db.query(Run).filter(
                    Run.user_id == user_id,
                    Run.target_id == target_ids[label],
                    Run.suite_id == suite_id,
                ).one_or_none()
                if existing is not None:
                    continue
                run = Run(
                    user_id=user_id,
                    target_id=target_ids[label],
                    suite_id=suite_id,
                    suite_version=str(suite.get("version")),
                    family=str(suite.get("family")),
                    status="queued",
                    sample_mode=True,
                    manifest_json=json.dumps(
                        {"suite_id": suite_id, "suite_version": suite.get("version"),
                         "scorer": suite.get("scorer"), "criteria": suite.get("criteria"),
                         "model": label, "sample_mode": True, "seed": 20260921,
                         "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024,
                         "note": "demo data — no model was called"}
                    ),
                )
                db.add(run)
                db.flush()
                run_id = run.id

            original = runner.get_adapter
            runner.get_adapter = lambda _provider, _m=model: _m
            try:
                asyncio.run(runner.execute_run(run_id))
            finally:
                runner.get_adapter = original
            print(f"seeded {suite_id} for {label}")

    print(f"\nSign in as {DEMO_EMAIL} / {DEMO_PASSWORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
