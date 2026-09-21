"""Run execution: turn a suite plus a user's model target into scored evidence.

A run is a manifest (suite + version + target + sampling parameters + dataset
revision), and the same manifest reproduces the same item set (REQ-EX-01/02).
Prompts and responses are persisted as evidence for every item (REQ-EX-07).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from harness.loader import packs_dir

from . import scoring
from .config import get_settings
from .db import session_scope
from .models import ModelTarget, Result, Run, utcnow
from .packs import current
from .providers import ProviderError, get_adapter
from .providers.base import GenerationParams
from .security import DecryptionError, decrypt_secret, redact

log = logging.getLogger("aish.runner")

EU_MMLU_DATA = Path("data/eu_mmlu.jsonl")


class RunError(Exception):
    """A run could not be built or executed. Message is safe to show a user."""


@dataclass
class Item:
    item_id: str
    prompt: str
    stratum: str = ""
    variant: str = ""
    system: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Item construction
# --------------------------------------------------------------------------- #


def _eu_mmlu_rows() -> list[dict]:
    path = EU_MMLU_DATA
    if not path.exists():
        raise RunError(
            "The EU-MMLU dataset is not provisioned on this server. "
            "Run scripts/fetch_eu_mmlu.py to download it, then retry."
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RunError("The EU-MMLU dataset file is present but empty.")
    return rows


def _build_eu_mmlu(suite: dict, sample: bool, seed: int) -> list[Item]:
    rows = _eu_mmlu_rows()
    languages = {lang.lower() for lang in suite.get("languages") or []}
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        language = str(row.get("Language", "")).lower()
        if languages and language not in languages:
            continue
        by_stratum[(language, str(row.get("Subject", "")))].append(row)

    if not by_stratum:
        raise RunError("No EU-MMLU rows matched this suite's language selection.")

    per_stratum = (suite.get("sampling_mode") or {}).get("items_per_stratum", 25)
    # Seeded, reproducible item selection — deliberately not a cryptographic RNG.
    rng = random.Random(seed)  # nosec B311
    items: list[Item] = []
    template = (suite.get("prompt") or {}).get("template", "")

    for (language, subject), stratum_rows in sorted(by_stratum.items()):
        chosen = stratum_rows
        if sample:
            chosen = sorted(stratum_rows, key=lambda r: r.get("Index", 0))
            if len(chosen) > per_stratum:
                chosen = rng.sample(chosen, per_stratum)
        for row in chosen:
            choices = [str(row.get(f"Choice_{i}", "")) for i in range(4)]
            prompt = template.format(
                question=row.get("Question", ""),
                choice_0=choices[0], choice_1=choices[1],
                choice_2=choices[2], choice_3=choices[3],
            )
            items.append(
                Item(
                    item_id=f"{language}/{subject}/{row.get('Index')}",
                    prompt=prompt,
                    stratum=f"{language}|{subject}",
                    variant=language,
                    meta={"choices": choices, "answer": row.get("Answer"), "language": language,
                          "subject": subject},
                )
            )
    return items


def _resolve_variants(view, item: dict) -> list[str]:
    if item.get("variants"):
        return [str(v) for v in item["variants"]]
    ref = item.get("variants_from")
    if not ref:
        return []
    attribute_id = ref.split(".", 1)[-1]
    for attribute in view.attributes():
        if attribute.get("id") == attribute_id:
            return [str(v) for v in attribute.get("variants") or []]
    return []


def _build_bias(suite: dict, view, sample: bool, seed: int) -> list[Item]:
    sampling = suite.get("sampling") or {}
    samples = 1 if sample else int(sampling.get("samples_per_variant", 1))
    # Seeded, reproducible variant selection — deliberately not a cryptographic RNG.
    rng = random.Random(seed)  # nosec B311
    items: list[Item] = []

    for probe in suite.get("items") or []:
        axis = probe.get("axis", "")
        variants = _resolve_variants(view, probe)
        if sample and len(variants) > 8:
            variants = rng.sample(variants, 8)
        template = probe.get("template", "")
        name = (probe.get("name_control") or {}).get("value", "")
        for variant in variants:
            for replicate in range(samples):
                prompt = template.replace("{" + axis + "}", variant).replace("{name}", name)
                items.append(
                    Item(
                        item_id=f"{probe['id']}#{variant}#{replicate}",
                        prompt=prompt,
                        stratum=probe["id"],
                        variant=variant,
                        meta={
                            "bias_type": probe.get("bias_type"),
                            "axis": axis,
                            "metric": probe.get("metric"),
                            "outcome_extraction": probe.get("outcome_extraction") or {},
                            "probe_id": probe["id"],
                        },
                    )
                )
    if not items:
        raise RunError("This bias suite defines no expandable templates.")
    return items


def _build_grounding(suite: dict) -> list[Item]:
    items: list[Item] = []
    for probe in suite.get("items") or []:
        prompt = probe.get("prompt", "")
        source_ref = probe.get("source")
        if source_ref:
            source_path = packs_dir() / source_ref
            if not source_path.exists():
                raise RunError(f"Grounding source file is missing: {source_ref}")
            source_text = source_path.read_text(encoding="utf-8")
            prompt = f"Supplied text:\n\n{source_text}\n\n---\n\n{prompt}"
        items.append(
            Item(
                item_id=probe["id"],
                prompt=prompt,
                stratum=probe.get("domain", ""),
                variant=probe.get("item_type", ""),
                meta={
                    "expected": probe.get("expected"),
                    "reference_answer": probe.get("reference_answer"),
                    "item_type": probe.get("item_type"),
                },
            )
        )
    if not items:
        raise RunError("This grounding suite defines no items.")
    return items


def build_items(suite: dict, view, *, sample: bool, seed: int) -> list[Item]:
    family = suite.get("family")
    if family == "capability":
        return _build_eu_mmlu(suite, sample, seed)
    if family == "bias":
        return _build_bias(suite, view, sample, seed)
    if family == "grounding":
        return _build_grounding(suite)
    if family == "safety":
        raise RunError(
            "The AILuminate prompt set is not redistributed with this harness and is "
            "not provisioned on this server. Provision it under data/ and pin its "
            "revision in packs/suites/ailuminate.yaml before running this suite."
        )
    raise RunError(f"No executor is registered for suite family {family!r}.")


# --------------------------------------------------------------------------- #
# Scoring one response
# --------------------------------------------------------------------------- #


def score_item(family: str, item: Item, text: str, refused: bool) -> tuple[str, float | None]:
    if refused:
        return "refused", None

    if family == "capability":
        parsed = scoring.parse_choice(text, item.meta.get("choices") or [])
        if parsed.index is None:
            return scoring.UNPARSEABLE, None
        correct = parsed.index == item.meta.get("answer")
        return ("correct" if correct else "incorrect"), (1.0 if correct else 0.0)

    if family == "grounding":
        outcome = scoring.classify_grounding(text, item.meta)
        positive = outcome in (scoring.GROUNDED, scoring.ABSTAINED_CORRECTLY)
        return outcome, (1.0 if positive else 0.0)

    if family == "bias":
        extraction = item.meta.get("outcome_extraction") or {}
        if extraction:
            outcome = scoring.extract_outcome(text, extraction)
            if outcome is None:
                return scoring.UNPARSEABLE, None
            return outcome, (1.0 if outcome == "favourable" else 0.0)
        if scoring.is_refusal(text):
            return "refused_to_answer", 0.0
        return "measured", scoring.information_content(text)

    return scoring.UNPARSEABLE, None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def summarise(suite: dict, view, rows: list[Result]) -> dict:
    family = suite.get("family")
    criteria = view.criteria().get(suite.get("criteria"), {})
    bands = criteria.get("metrics") or {}
    summary: dict = {
        "family": family,
        "criteria": suite.get("criteria"),
        "n": len(rows),
        "errors": sum(1 for r in rows if r.error),
        "refused": sum(1 for r in rows if r.outcome == "refused"),
        "metrics": {},
        "strata": [],
        "verdicts": {},
    }
    if not rows:
        return summary

    if family == "capability":
        summary.update(_summarise_capability(rows, suite))
    elif family == "grounding":
        summary.update(_summarise_grounding(rows))
    elif family == "bias":
        summary.update(_summarise_bias(rows, criteria))

    summary["verdicts"] = {
        name: scoring.verdict(
            summary["metrics"].get(name),
            band,
            summary.get("intervals", {}).get(name),
        )
        for name, band in bands.items()
        if name in summary["metrics"]
    }
    summary["verdict"] = scoring.worst_verdict(list(summary["verdicts"].values()))
    return summary


def _rate(rows: list[Result], predicate) -> tuple[float | None, int, int]:
    total = len(rows)
    if total == 0:
        return None, 0, 0
    hits = sum(1 for r in rows if predicate(r))
    return hits / total, hits, total


def _summarise_capability(rows: list[Result], suite: dict) -> dict:
    scored = [r for r in rows if r.outcome in ("correct", "incorrect")]
    unparsed = sum(1 for r in rows if r.outcome == scoring.UNPARSEABLE)
    accuracy = (sum(1 for r in scored if r.outcome == "correct") / len(scored)) if scored else None

    by_language: dict[str, list[Result]] = defaultdict(list)
    for row in rows:
        by_language[row.variant].append(row)

    reference = (suite.get("reference_language") or "en").lower()
    language_accuracy: dict[str, float] = {}
    for language, language_rows in by_language.items():
        judged = [r for r in language_rows if r.outcome in ("correct", "incorrect")]
        if judged:
            language_accuracy[language] = sum(
                1 for r in judged if r.outcome == "correct"
            ) / len(judged)

    baseline = language_accuracy.get(reference)
    parity = (
        {lang: (acc / baseline) for lang, acc in language_accuracy.items() if lang != reference}
        if baseline
        else {}
    )

    strata = []
    for language in sorted(by_language):
        judged = [r for r in by_language[language] if r.outcome in ("correct", "incorrect")]
        correct = sum(1 for r in judged if r.outcome == "correct")
        low, high = scoring.wilson_interval(correct, len(judged)) if judged else (0.0, 1.0)
        strata.append(
            {
                "key": language,
                "n": len(by_language[language]),
                "value": (correct / len(judged)) if judged else None,
                "ci": [low, high],
                "parity": parity.get(language),
            }
        )

    metrics = {
        "accuracy": accuracy,
        "unparsed_rate": unparsed / len(rows),
    }
    if parity:
        metrics["parity_floor"] = min(parity.values())
        metrics["parity_spread"] = max(parity.values()) - min(parity.values())

    correct_count = sum(1 for r in scored if r.outcome == "correct")
    return {
        "metrics": metrics,
        "intervals": {"accuracy": list(scoring.wilson_interval(correct_count, len(scored)))}
        if scored
        else {},
        "strata": strata,
        "reference_language": reference,
        "language_accuracy": language_accuracy,
        "parity": parity,
    }


def _summarise_grounding(rows: list[Result]) -> dict:
    counts = defaultdict(int)
    for row in rows:
        counts[row.outcome] += 1
    total = len(rows)

    answerable = [r for r in rows if r.variant != "unanswerable"]
    unanswerable = [r for r in rows if r.variant == "unanswerable"]
    abstentions = [
        r for r in rows
        if r.outcome in (scoring.ABSTAINED_CORRECTLY, scoring.ABSTAINED_WRONGLY)
    ]

    grounded_rate = (
        sum(1 for r in answerable if r.outcome == scoring.GROUNDED) / len(answerable)
        if answerable else None
    )
    abstention_precision = (
        sum(1 for r in abstentions if r.outcome == scoring.ABSTAINED_CORRECTLY) / len(abstentions)
        if abstentions else None
    )
    correct_abstention = (
        sum(1 for r in unanswerable if r.outcome == scoring.ABSTAINED_CORRECTLY) / len(unanswerable)
        if unanswerable else 0.0
    )
    wrong_abstention = (
        sum(1 for r in answerable if r.outcome == scoring.ABSTAINED_WRONGLY) / len(answerable)
        if answerable else 0.0
    )

    return {
        "metrics": {
            "hallucination_rate": counts[scoring.HALLUCINATED] / total,
            "grounded_rate": grounded_rate,
            "abstention_precision": abstention_precision,
            "calibration": correct_abstention - wrong_abstention,
        },
        "strata": [
            {"key": outcome, "n": count, "value": count / total, "ci": None}
            for outcome, count in sorted(counts.items())
        ],
        "outcome_counts": dict(counts),
    }


def _summarise_bias(rows: list[Result], criteria: dict) -> dict:
    by_probe: dict[str, list[Result]] = defaultdict(list)
    for row in rows:
        by_probe[row.stratum].append(row)

    confidence = criteria.get("confidence") or {}
    resamples = int(confidence.get("resamples", 2000))
    # The pack declares 10,000 resamples; the interactive path caps this so a run
    # page stays responsive. The cap is reported, never silently applied.
    capped = min(resamples, 2000)

    groups = []
    rate_disparities: list[float] = []
    refusal_disparities: list[float] = []
    quality_disparities: list[float] = []
    intervals: dict[str, list[float]] = {}

    for probe_id, probe_rows in sorted(by_probe.items()):
        by_variant: dict[str, list[Result]] = defaultdict(list)
        for row in probe_rows:
            by_variant[row.variant].append(row)

        has_outcomes = any(r.outcome in ("favourable", "unfavourable") for r in probe_rows)
        per_variant_rate: dict[str, float] = {}
        per_variant_samples: dict[str, list[float]] = {}
        per_variant_refusal: dict[str, float] = {}

        for variant, variant_rows in by_variant.items():
            judged = [r for r in variant_rows if r.outcome in ("favourable", "unfavourable")]
            if has_outcomes and judged:
                values = [1.0 if r.outcome == "favourable" else 0.0 for r in judged]
            else:
                raw = [r.score for r in variant_rows if r.score is not None]
                peak = max(raw) if raw else 0.0
                values = [(v / peak) if peak else 0.0 for v in raw]
            if values:
                per_variant_rate[variant] = sum(values) / len(values)
                per_variant_samples[variant] = values
            refused = [r for r in variant_rows if r.outcome in ("refused", "refused_to_answer")]
            per_variant_refusal[variant] = len(refused) / len(variant_rows) if variant_rows else 0.0

        spread = scoring.disparity(per_variant_rate)
        interval = scoring.bootstrap_disparity_ci(per_variant_samples, resamples=capped)
        refusal_spread = scoring.disparity(per_variant_refusal)

        if has_outcomes:
            rate_disparities.append(spread)
            metric_name = "rate_disparity"
        else:
            quality_disparities.append(spread)
            metric_name = "quality_disparity"
        refusal_disparities.append(refusal_spread)

        groups.append(
            {
                "probe_id": probe_id,
                "metric": metric_name,
                "disparity": spread,
                "ci": list(interval),
                "refusal_disparity": refusal_spread,
                "axis": probe_rows[0].item_id.split("#")[0] if probe_rows else "",
                "variants": [
                    {
                        "variant": variant,
                        "rate": per_variant_rate.get(variant),
                        "refusal": per_variant_refusal.get(variant, 0.0),
                        "n": len(by_variant[variant]),
                    }
                    for variant in sorted(by_variant)
                ],
            }
        )

    metrics: dict[str, float | None] = {}
    if rate_disparities:
        metrics["rate_disparity"] = max(rate_disparities)
        worst = max(groups, key=lambda g: g["disparity"] if g["metric"] == "rate_disparity" else -1)
        intervals["rate_disparity"] = worst["ci"]
    if refusal_disparities:
        metrics["refusal_disparity"] = max(refusal_disparities)
    if quality_disparities:
        metrics["quality_disparity"] = max(quality_disparities)

    return {
        "metrics": metrics,
        "intervals": intervals,
        "strata": [
            {"key": g["probe_id"], "n": sum(v["n"] for v in g["variants"]),
             "value": g["disparity"], "ci": g["ci"]}
            for g in groups
        ],
        "groups": groups,
        "bootstrap_resamples": capped,
        "bootstrap_capped_from": resamples if capped != resamples else None,
    }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

_active: set[int] = set()


def active_runs() -> set[int]:
    return set(_active)


async def execute_run(run_id: int) -> None:
    """Execute a queued run. Safe to call once per run; re-entry is a no-op."""
    if run_id in _active:
        return
    _active.add(run_id)
    try:
        await _execute(run_id)
    except Exception as exc:  # last-resort guard: a crashed task must mark the run
        log.exception("run %s crashed", run_id)
        _fail(run_id, str(exc))
    finally:
        _active.discard(run_id)


def _fail(run_id: int, message: str) -> None:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run and run.status in ("queued", "running"):
            run.status = "failed"
            run.error = message[:600]
            run.finished_at = utcnow()


async def _execute(run_id: int) -> None:
    settings = get_settings()

    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None or run.status != "queued":
            return
        target = session.get(ModelTarget, run.target_id)
        if target is None:
            raise RunError("The model target for this run no longer exists.")
        manifest = json.loads(run.manifest_json or "{}")
        suite_id, sample, seed = run.suite_id, run.sample_mode, int(manifest.get("seed", 20260921))
        provider, model_name, endpoint = target.provider, target.model_name, target.endpoint
        ciphertext, owner = target.api_key_ciphertext, f"user:{target.user_id}"

    view = current()
    suite = view.suite(suite_id)
    if suite is None:
        raise RunError(f"Suite {suite_id!r} is no longer defined.")

    items = build_items(suite, view, sample=sample, seed=seed)
    if len(items) > settings.max_items_per_run:
        # Truncating here would drop whole strata from the tail of an ordered item
        # list and still report a parity floor over the survivors. Refuse instead.
        raise RunError(
            f"This run would send {len(items)} items, above this instance's limit of "
            f"{settings.max_items_per_run}. Raise AISH_MAX_ITEMS_PER_RUN or use "
            "sample mode."
        )

    api_key = None
    if ciphertext:
        try:
            api_key = decrypt_secret(ciphertext, owner)
        except DecryptionError as exc:
            raise RunError("The stored API key for this target could not be decrypted.") from exc

    with session_scope() as session:
        run = session.get(Run, run_id)
        run.status = "running"
        run.started_at = utcnow()
        run.items_total = len(items)

    adapter = get_adapter(provider)
    params = GenerationParams(
        temperature=float(manifest.get("temperature", 0.0)),
        top_p=float(manifest.get("top_p", 1.0)),
        max_tokens=int(manifest.get("max_tokens", 1024)),
        seed=seed,
    )
    family = suite.get("family")
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    completed = 0
    lock = asyncio.Lock()

    async def run_one(item: Item) -> None:
        nonlocal completed
        async with semaphore:
            outcome, score, text, error = "", None, "", ""
            latency = tokens_in = tokens_out = 0
            try:
                completion = await adapter.generate(
                    model=model_name,
                    prompt=item.prompt,
                    system=item.system,
                    api_key=api_key,
                    endpoint=endpoint or None,
                    params=params,
                    timeout=settings.request_timeout_seconds,
                )
                text = completion.text
                latency = completion.latency_ms
                tokens_in, tokens_out = completion.tokens_in, completion.tokens_out
                outcome, score = score_item(family, item, text, completion.refused)
            except ProviderError as exc:
                # Transport failures are their own category, never a wrong answer.
                outcome = "error"
                error = redact(str(exc), api_key or "")[:400]
            except Exception as exc:  # noqa: BLE001 - one bad item must not kill the run
                outcome = "error"
                error = redact(f"{type(exc).__name__}: {exc}", api_key or "")[:400]

        with session_scope() as session:
            session.add(
                Result(
                    run_id=run_id,
                    item_id=item.item_id,
                    stratum=item.stratum,
                    variant=item.variant,
                    prompt=item.prompt[:20000],
                    response=text[:20000],
                    outcome=outcome,
                    score=score,
                    latency_ms=latency,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    error=error,
                )
            )
        async with lock:
            completed += 1
            if completed % 5 == 0 or completed == len(items):
                with session_scope() as session:
                    current_run = session.get(Run, run_id)
                    if current_run:
                        current_run.items_done = completed

    await asyncio.gather(*(run_one(item) for item in items))

    with session_scope() as session:
        run = session.get(Run, run_id)
        rows = list(session.scalars(select(Result).where(Result.run_id == run_id)))
        summary = summarise(suite, view, rows)
        summary["sample_mode"] = sample
        summary["scorer"] = suite.get("scorer")
        summary["suite_version"] = suite.get("version")
        run.summary_json = json.dumps(summary)
        run.items_done = len(rows)
        run.status = "done"
        run.finished_at = utcnow()
        errors = sum(1 for r in rows if r.outcome == "error")
        if errors == len(rows) and rows:
            run.status = "failed"
            run.error = rows[0].error or "every request to the model failed"


# The event loop the application runs on, captured at start-up.
#
# A sync FastAPI route executes in an AnyIO worker thread, which has no event loop
# of its own, so a run cannot be scheduled from there without a reference to the
# loop that does. Holding one here is what lets a synchronous route hand work to the
# loop and return immediately.
_loop: asyncio.AbstractEventLoop | None = None

# asyncio keeps only weak references to tasks, so a task with no other reference can
# be garbage collected mid-run. These references are what keep a run alive.
_pending: set = set()


def bind_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


class RunNotScheduled(RunError):
    """The run row exists but could not be handed to the event loop."""


def queue_run(run_id: int) -> None:
    """Schedule a run, from either an async or a synchronous caller."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:  # already on the loop
        task = loop.create_task(execute_run(run_id))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
        return

    if _loop is None or _loop.is_closed():
        raise RunNotScheduled(
            "The application is not ready to execute runs. Try again in a moment."
        )

    # Called from a worker thread: hand the coroutine to the application's loop.
    future = asyncio.run_coroutine_threadsafe(execute_run(run_id), _loop)
    _pending.add(future)
    future.add_done_callback(_pending.discard)


def utc_iso(value: datetime | None) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds") if value else ""
