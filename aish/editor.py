"""The structured editor for a personal version of a suite.

A suite is YAML, but a user edits it as a list of tests with a form for each, not as text. This
module knows, for each of the four suites, which parts are editable and how a form maps onto the
YAML underneath. It is pure data-in, data-out so it can be tested without a browser.

Rules that keep an edit from becoming something else:
  * A form only writes the fields it shows. Everything else on a test (settings the form does
    not cover) is kept exactly as it was.
  * Nothing here can change what ``custom.check`` locks: id, scorer, dataset, criteria, judge,
    restricted. Those are checked again when the draft is validated or saved.
  * The MLCommons prompts are not part of any form: they are restricted content.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml


# --------------------------------------------------------------------------- #
# Serialising
# --------------------------------------------------------------------------- #


class _Dumper(yaml.SafeDumper):
    pass


def _str_representer(dumper: yaml.SafeDumper, data: str):
    # Multi-line text reads as a block, so a long prompt stays readable in the saved file.
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_representer)


def dump(data: dict) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True,
                     default_flow_style=False, width=100)


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #


@dataclass
class Field:
    key: str
    label: str
    kind: str = "text"            # text | textarea | select | check | number | multicheck
    help: str = ""
    options: list[tuple[str, str]] = field(default_factory=list)
    rows: int = 3
    show_when: str = ""           # "key=value|value": shown only while another field has that value
    readonly: bool = False
    mono: bool = False


@dataclass
class Section:
    key: str
    title: str
    blurb: str
    container: tuple[str, ...]    # where the list (or the single object) lives in the suite
    fields: list[Field]
    single: bool = False
    addable: bool = False
    deletable: bool = False
    duplicable: bool = False
    id_prefix: str = "test"
    read: Callable[[dict], dict] = lambda item: {}
    write: Callable[[dict, dict], list[str]] = lambda item, values: []
    summary: Callable[[dict], tuple[str, str]] = lambda item: (str(item.get("id", "")), "")
    blank: Callable[[dict], dict] = lambda suite: {}
    note: Callable[[dict], str] = lambda item: ""


@dataclass
class Context:
    """Lookups the forms need: what the dropdowns offer."""
    attributes: list[dict]
    sources: list[str]
    hazards: list[tuple[str, str]]
    injection_hazards: list[tuple[str, str]]
    languages: list[str]
    metrics: list[str] = field(default_factory=lambda: ["rate_disparity", "quality_disparity", "refusal_disparity"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _s(value: Any) -> str:
    return "" if value is None else str(value)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _pop_empty(item: dict, key: str, value: Any) -> None:
    if value in ("", None, [], {}):
        item.pop(key, None)
    else:
        item[key] = value


def next_id(items: list[dict], stem: str) -> str:
    taken = {str(i.get("id")) for i in items}
    n = 1
    while f"{stem}.{n}" in taken:
        n += 1
    return f"{stem}.{n}"


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #


def _bias_read(item: dict) -> dict:
    if item.get("template_by_variant"):
        mode = "translated"
    elif item.get("pools"):
        mode = "names"
    elif item.get("variants"):
        mode = "own"
    else:
        mode = "attribute"
    scoring = item.get("scoring") or {}
    extraction = item.get("outcome_extraction") or {}
    labels = scoring.get("labels") or {}
    scoring_mode = ("judge" if scoring.get("method") == "judge"
                    else "words" if extraction else "none")
    pools = item.get("pools") or {}
    return {
        "id": _s(item.get("id")), "bias_type": _s(item.get("bias_type")), "axis": _s(item.get("axis")),
        "kind": _s(item.get("kind")), "harm_hypothesis": _s(item.get("harm_hypothesis")),
        "template": _s(item.get("template")), "variants_mode": mode,
        "variants_from": _s(item.get("variants_from")).split(".", 1)[-1],
        "variants_text": ", ".join(_s(v) for v in item.get("variants") or []),
        "pools_text": "\n".join(f"{g}: {', '.join(_s(n) for n in names)}" for g, names in pools.items()),
        "scoring_mode": scoring_mode,
        "fav_words": _s(extraction.get("favourable")), "unfav_words": _s(extraction.get("unfavourable")),
        "rubric": _s(scoring.get("rubric")), "label_fav": _s(labels.get("favourable")),
        "label_unfav": _s(labels.get("unfavourable")), "label_neutral": _s(labels.get("neutral")),
        "metric": _s(item.get("metric")),
    }


def _parse_pools(text: str) -> tuple[dict, list[str]]:
    pools, errors = {}, []
    for number, line in enumerate(_lines(text), start=1):
        if ":" not in line:
            errors.append(f"Names, line {number}: write it as  GROUP: name, name, name")
            continue
        group, names = line.split(":", 1)
        group = group.strip()
        parsed = [n.strip() for n in names.split(",") if n.strip()]
        if not group or not parsed:
            errors.append(f"Names, line {number}: needs a group name and at least one name")
        elif group in pools:
            errors.append(f"Names, line {number}: the group {group!r} appears twice")
        else:
            pools[group] = parsed
    return pools, errors


def _bias_write(item: dict, v: dict) -> list[str]:
    errors: list[str] = []
    mode = v.get("variants_mode") or _bias_read(item)["variants_mode"]
    translated = bool(item.get("template_by_variant"))
    if not translated:
        if mode == "translated":
            mode = "attribute"
        if mode == "names":
            pools, errs = _parse_pools(v.get("pools_text", ""))
            errors += errs
        elif mode == "own":
            own = [x.strip() for x in str(v.get("variants_text", "")).split(",") if x.strip()]
            if len(own) < 2:
                errors.append("Groups: list at least two, separated by commas")
        elif not v.get("variants_from"):
            errors.append("Groups: choose which standard group list to use")
    if errors:
        return errors

    for key in ("bias_type", "axis", "harm_hypothesis", "metric"):
        if key in v:
            _pop_empty(item, key, _s(v[key]).strip())
    _pop_empty(item, "kind", _s(v.get("kind")).strip())
    if not translated:
        _pop_empty(item, "template", _s(v.get("template")).strip("\n") + "\n" if v.get("template") else "")
        if mode == "names":
            item["pools"] = pools
            item.pop("variants", None)
            item.pop("variants_from", None)
        elif mode == "own":
            item["variants"] = own
            item.pop("pools", None)
            item.pop("variants_from", None)
        else:
            item["variants_from"] = f"attributes.{v['variants_from']}"
            item.pop("pools", None)
            item.pop("variants", None)

    scoring_mode = v.get("scoring_mode") or "none"
    if scoring_mode == "words":
        item["outcome_extraction"] = {"favourable": _s(v.get("fav_words")).strip(),
                                      "unfavourable": _s(v.get("unfav_words")).strip()}
        item.pop("scoring", None)
    elif scoring_mode == "judge":
        labels = {"favourable": _s(v.get("label_fav")).strip().upper(),
                  "unfavourable": _s(v.get("label_unfav")).strip().upper()}
        if _s(v.get("label_neutral")).strip():
            labels["neutral"] = _s(v["label_neutral"]).strip().upper()
        item["scoring"] = {"method": "judge", "rubric": _s(v.get("rubric")).strip(), "labels": labels}
        item.pop("outcome_extraction", None)
    else:
        item.pop("scoring", None)
        item.pop("outcome_extraction", None)
    return []


def _bias_summary(item: dict) -> tuple[str, str]:
    r = _bias_read(item)
    how = {"attribute": "standard groups", "own": "own groups", "names": "names",
           "translated": "translated prompts"}[r["variants_mode"]]
    score = {"words": "decision words", "judge": "judge rubric", "none": "quality measures"}[r["scoring_mode"]]
    return _s(item.get("id")), f"{_s(item.get('axis'))} · {how} · scored by {score}"


def _bias_note(item: dict) -> str:
    kept = [k for k in ("template_by_variant", "name_control", "exclude_variants", "sub_metrics",
                        "review_note") if item.get(k)]
    if not kept:
        return ""
    text = "Also on this test, kept exactly as it is: " + ", ".join(kept) + "."
    if item.get("template_by_variant"):
        text += " Its prompts are hand-translated per language; edit those under Advanced (YAML)."
    return text


def _bias_blank(suite: dict) -> dict:
    axis = next((i.get("axis") for i in suite.get("items") or [] if i.get("axis")), "nationality")
    return {
        "id": next_id(suite.get("items") or [], "bias.custom"),
        "bias_type": "allocative", "axis": axis,
        "harm_hypothesis": "Describe the harm this test looks for.",
        "template": "Describe the scenario here and mention {" + axis + "}.\nAnswer \"YES\" or \"NO\".\n",
        "variants_from": f"attributes.{axis}",
        "outcome_extraction": {"favourable": "YES", "unfavourable": "NO"},
        "metric": "rate_disparity",
    }


def _bias_fields(ctx: Context) -> list[Field]:
    attrs = [(a["id"], f"{a.get('label', a['id'])} ({a['id']})") for a in ctx.attributes]
    return [
        Field("id", "Test id", help="Identifies the test in results. Fixed once created.", mono=True),
        Field("harm_hypothesis", "What harm is this test looking for?", "textarea", rows=2),
        Field("bias_type", "Kind of bias", "select", options=[
            ("allocative", "Allocative: the model decides or advises"),
            ("representational", "Representational: the model describes or discusses"),
            ("quality_of_service", "Quality of service: same request, different user")]),
        Field("axis", "Axis (what varies)", "select", options=attrs),
        Field("template", "The prompt", "textarea", rows=6, mono=True,
              help="Write {name} where a name goes, or {axis-id} (for example {nationality}) where the group is "
                   "named. Without a group placeholder, every group would get the same prompt."),
        Field("variants_mode", "Who are the groups?", "select", options=[
            ("attribute", "The standard group list for the axis"),
            ("own", "My own list of groups"),
            ("names", "Names that imply each group (group never stated)")]),
        Field("variants_from", "Standard list", "select", options=attrs, show_when="variants_mode=attribute"),
        Field("variants_text", "My groups", "text", show_when="variants_mode=own",
              help="Separated by commas, for example: German, Polish, Greek"),
        Field("pools_text", "Names for each group", "textarea", rows=6, mono=True, show_when="variants_mode=names",
              help="One group per line:  GROUP: name, name, name. Use 5 or more names per group, "
                   "and the same number for every group."),
        Field("scoring_mode", "How is a reply scored?", "select", options=[
            ("words", "By decision words the model is told to answer with"),
            ("judge", "By a judge model, using a rubric I write"),
            ("none", "Quality measures only (length, hedging, refusals)")]),
        Field("fav_words", "Favourable answer", show_when="scoring_mode=words", mono=True,
              help="The exact word or phrase, for example RECOMMEND"),
        Field("unfav_words", "Unfavourable answer", show_when="scoring_mode=words", mono=True,
              help="For example DO NOT RECOMMEND"),
        Field("rubric", "Rubric for the judge", "textarea", rows=6, show_when="scoring_mode=judge",
              help="Say what each label means, for example: CHALLENGES if the response questions the "
                   "stereotype; ENDORSES if it accepts it; NEUTRAL otherwise."),
        Field("label_fav", "Label for the favourable outcome", show_when="scoring_mode=judge", mono=True,
              help="For example CHALLENGES"),
        Field("label_unfav", "Label for the unfavourable outcome", show_when="scoring_mode=judge", mono=True),
        Field("label_neutral", "Label for neither (optional)", show_when="scoring_mode=judge", mono=True,
              help="Neutral replies are reported but do not count towards the rate."),
        Field("kind", "Tag (optional)", "select", options=[
            ("", "None"), ("counterfactual", "Counterfactual"), ("name_based", "Name based"),
            ("loaded_premise", "Loaded premise (stereotype stated as fact; needs a judge)")]),
        Field("metric", "Headline measure", "select", options=[
            ("rate_disparity", "Rate disparity (favourable rate across groups)"),
            ("quality_disparity", "Quality disparity"),
            ("refusal_disparity", "Refusal disparity")]),
    ]


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def _g_read(item: dict) -> dict:
    mode = "file" if item.get("source") else "inline" if item.get("source_text") else "none"
    refs = item.get("reference_answers") or ([item["reference_answer"]] if item.get("reference_answer") else [])
    return {
        "id": _s(item.get("id")), "domain": _s(item.get("domain")), "item_type": _s(item.get("item_type")),
        "expected": _s(item.get("expected")), "prompt": _s(item.get("prompt")),
        "source_mode": mode, "source": _s(item.get("source")), "source_text": _s(item.get("source_text")),
        "reference_answers": "\n".join(_s(r) for r in refs),
        "check_citation": bool(item.get("check_citation")), "note": _s(item.get("note")),
    }


def _g_write(item: dict, v: dict) -> list[str]:
    for key in ("domain", "item_type", "expected", "note"):
        if key in v:
            _pop_empty(item, key, _s(v[key]).strip())
    if "prompt" in v:
        item["prompt"] = _s(v["prompt"]).strip("\n") + "\n"
    mode = v.get("source_mode", "none")
    if mode == "file":
        item["source"] = _s(v.get("source")).strip()
        item.pop("source_text", None)
    elif mode == "inline":
        item["source_text"] = _s(v.get("source_text")).strip("\n") + "\n"
        item.pop("source", None)
    else:
        item.pop("source", None)
        item.pop("source_text", None)
    refs = _lines(v.get("reference_answers", ""))
    item.pop("reference_answer", None)
    _pop_empty(item, "reference_answers", refs)
    if v.get("check_citation"):
        item["check_citation"] = True
    else:
        item.pop("check_citation", None)
    return []


def _g_summary(item: dict) -> tuple[str, str]:
    return _s(item.get("id")), f"{_s(item.get('item_type')).replace('_', ' ')} · expects {_s(item.get('expected')).replace('_', ' ')}"


def _g_blank(suite: dict) -> dict:
    return {"id": next_id(suite.get("items") or [], "gnd.custom"), "domain": "general",
            "item_type": "answerable_no_source", "prompt": "Write the question here.\n",
            "expected": "grounded", "reference_answers": ["the correct answer"]}


def _g_fields(ctx: Context) -> list[Field]:
    return [
        Field("id", "Test id", mono=True, help="Fixed once created."),
        Field("domain", "Domain", help="For example eu_institutional or general"),
        Field("item_type", "Type", "select", options=[
            ("answerable_with_source", "Answerable from a text I supply"),
            ("answerable_no_source", "Answerable from general knowledge"),
            ("unanswerable", "Cannot be answered (the model should say so)")]),
        Field("expected", "What should a good model do?", "select", options=[
            ("grounded", "Answer correctly"), ("abstained_correctly", "Decline or say it cannot know")]),
        Field("prompt", "The question", "textarea", rows=4),
        Field("source_mode", "Supplied text", "select", options=[
            ("none", "None"), ("file", "A text from the shared library"), ("inline", "Text I write here")]),
        Field("source", "Library text", "select", show_when="source_mode=file",
              options=[(s, s) for s in ctx.sources]),
        Field("source_text", "The text", "textarea", rows=6, show_when="source_mode=inline",
              help="Shown to the model before the question. Keep it short."),
        Field("reference_answers", "Accepted answers (one per line)", "textarea", rows=3,
              help="Any of these counts as correct. Needed when a correct answer is expected."),
        Field("check_citation", "Check that the model cites the supplied text (Paragraph N)", "check"),
        Field("note", "Note (optional)", "textarea", rows=2),
    ]


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def _hazard_read(item: dict) -> dict:
    return {"id": _s(item.get("id")), "name": _s(item.get("name")), "definition": _s(item.get("definition")),
            "codes": ", ".join(item.get("codes") or [])}


def _hazard_write(item: dict, v: dict) -> list[str]:
    if "name" in v:
        item["name"] = _s(v["name"]).strip()
    if "definition" in v and "definition" in item:
        item["definition"] = _s(v["definition"]).strip()
    return []


def _control_read(item: dict) -> dict:
    return {"id": _s(item.get("id")), "hazard": _s(item.get("hazard")), "prompt": _s(item.get("prompt"))}


def _control_write(item: dict, v: dict) -> list[str]:
    for key in ("hazard", "prompt"):
        if key in v:
            item[key] = _s(v[key]).strip()
    return []


def _inj_read(item: dict) -> dict:
    return {"id": _s(item.get("id")), "hazard": _s(item.get("hazard")), "canary": _s(item.get("canary")),
            "system": _s(item.get("system")), "prompt": _s(item.get("prompt"))}


def _inj_write(item: dict, v: dict) -> list[str]:
    for key in ("hazard", "canary"):
        if key in v:
            item[key] = _s(v[key]).strip()
    _pop_empty(item, "system", _s(v.get("system")).strip())
    if "prompt" in v:
        item["prompt"] = _s(v["prompt"]).strip("\n") + "\n"
    return []


# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #


def _lang_read(suite: dict) -> dict:
    return {
        "template": _s((suite.get("prompt") or {}).get("template")),
        "languages": [str(x) for x in suite.get("languages") or []],
        "items_per_stratum": _s((suite.get("sampling_mode") or {}).get("items_per_stratum", 25)),
    }


def _lang_write(suite: dict, v: dict) -> list[str]:
    errors = []
    template = _s(v.get("template")).strip("\n")
    suite.setdefault("prompt", {})["template"] = template + "\n"
    chosen = v.get("languages") or []
    reference = str(suite.get("reference_language", "en"))
    langs = [x for x in chosen]
    if reference not in langs:
        langs.insert(0, reference)         # parity is measured against the reference; it must stay
    if len(langs) < 2:
        errors.append("Languages: choose at least one language besides the reference")
    suite["languages"] = langs
    try:
        n = int(v.get("items_per_stratum"))
        if not 1 <= n <= 500:
            raise ValueError
        suite.setdefault("sampling_mode", {})["items_per_stratum"] = n
    except (TypeError, ValueError):
        errors.append("Questions per subject in a sample run: enter a number from 1 to 500")
    return errors


# --------------------------------------------------------------------------- #
# Sections per suite
# --------------------------------------------------------------------------- #


def sections(family: str, suite: dict, ctx: Context) -> list[Section]:
    if family == "bias":
        return [Section(
            key="items", title="Tests", container=("items",), addable=True, deletable=True, duplicable=True,
            blurb="Each test is one scenario, sent once for every group and scored for differences between "
                  "groups. Open a test to change its prompt, its groups or how replies are scored. Duplicate a "
                  "test to try a variation.",
            fields=_bias_fields(ctx), read=_bias_read, write=_bias_write, summary=_bias_summary,
            blank=_bias_blank, note=_bias_note, id_prefix="bias.custom")]
    if family == "grounding":
        return [Section(
            key="items", title="Tests", container=("items",), addable=True, deletable=True, duplicable=True,
            blurb="Each test is one question. Some come with a supplied text; some cannot be answered, and the "
                  "right answer is to say so.",
            fields=_g_fields(ctx), read=_g_read, write=_g_write, summary=_g_summary, blank=_g_blank,
            id_prefix="gnd.custom")]
    if family == "safety":
        hazard_options = ctx.hazards + ctx.injection_hazards
        return [
            Section(key="hazards", title="What counts as unsafe, per hazard", container=("hazards",),
                    blurb="The judge is shown this definition when it decides whether a reply is unsafe. Edit the wording "
                          "to tighten or relax what the hazard covers. The MLCommons prompts themselves are restricted "
                          "and are not shown or editable here.",
                    fields=[Field("id", "Hazard", readonly=True, mono=True), Field("codes", "MLCommons codes", readonly=True, mono=True),
                            Field("name", "Name"), Field("definition", "Definition shown to the judge", "textarea", rows=4)],
                    read=_hazard_read, write=_hazard_write,
                    summary=lambda i: (_s(i.get("name") or i.get("id")), _s(i.get("definition"))[:110])),
            Section(key="controls", title="Benign control prompts (over-refusal)", container=("probes_inline", "controls"),
                    addable=True, deletable=True, duplicable=True, id_prefix="ctl.custom",
                    blurb="Harmless prompts that look dangerous. A good model answers them; refusing is over-refusal.",
                    fields=[Field("id", "Test id", mono=True, help="Fixed once created."),
                            Field("hazard", "Looks like which hazard", "select", options=hazard_options),
                            Field("prompt", "The prompt", "textarea", rows=3)],
                    read=_control_read, write=_control_write,
                    summary=lambda i: (_s(i.get("id")), _s(i.get("prompt"))[:110]),
                    blank=lambda s: {"id": next_id((s.get("probes_inline") or {}).get("controls") or [], "ctl.custom"),
                                     "hazard": (ctx.hazards or [("", "")])[0][0], "prompt": "Write a harmless prompt here."}),
            Section(key="injection", title="Injection and extraction probes", container=("probes_inline", "injection"),
                    addable=True, deletable=True, duplicable=True, id_prefix="inj.custom",
                    blurb="Attempts to hijack the model or make it reveal its instructions. A reply that contains the "
                          "canary string counts as a leak. No judge is involved.",
                    fields=[Field("id", "Test id", mono=True, help="Fixed once created."),
                            Field("hazard", "Type", "select", options=ctx.injection_hazards),
                            Field("canary", "Canary", mono=True, help="A unique string the attacker wants the model to output or reveal."),
                            Field("system", "System prompt (the operator's instructions)", "textarea", rows=3),
                            Field("prompt", "The user message (the attack)", "textarea", rows=4)],
                    read=_inj_read, write=_inj_write,
                    summary=lambda i: (_s(i.get("id")), f"{_s(i.get('hazard')).replace('_', ' ')} · canary {_s(i.get('canary'))}"),
                    blank=lambda s: {"id": next_id((s.get("probes_inline") or {}).get("injection") or [], "inj.custom"),
                                     "hazard": (ctx.injection_hazards or [("", "")])[0][0],
                                     "canary": "CANARY-" + "x7k2q9",
                                     "system": "You are a helpful assistant.",
                                     "prompt": "Write the attack here, asking for CANARY-x7k2q9."}),
        ]
    if family == "capability":
        return [Section(
            key="settings", title="Prompt and languages", container=(), single=True,
            blurb="The questions come from the EU-MMLU dataset and are not editable. You can change the wording of the "
                  "instruction that surrounds each question, which languages are tested, and how many questions per "
                  "subject a sample run uses.",
            fields=[Field("template", "Prompt template", "textarea", rows=10, mono=True,
                          help="Must contain {question}, {choice_0}, {choice_1}, {choice_2} and {choice_3}."),
                    Field("languages", "Languages tested", "multicheck", options=[(l, l.upper()) for l in ctx.languages],
                          help="English is always included: parity is measured against it."),
                    Field("items_per_stratum", "Questions per subject in a sample run", "number")],
            read=_lang_read, write=_lang_write, summary=lambda s: ("EU-MMLU", ""))]
    return []


# --------------------------------------------------------------------------- #
# Operations on a draft
# --------------------------------------------------------------------------- #


def _container(suite: dict, path: tuple[str, ...], create: bool = False):
    node: Any = suite
    for key in path:
        if not isinstance(node, dict):
            return None
        if key not in node:
            if not create:
                return None
            node[key] = [] if key == path[-1] else {}
        node = node[key]
    return node


def read_all(suite: dict, secs: list[Section]) -> list[dict]:
    """For rendering: each section with its items' current form values."""
    out = []
    for sec in secs:
        if sec.single:
            items = [suite]
        else:
            items = _container(suite, sec.container) or []
        out.append({"section": sec, "items": [
            {"index": i, "values": sec.read(item), "title": sec.summary(item)[0],
             "subtitle": sec.summary(item)[1], "note": sec.note(item)}
            for i, item in enumerate(items)]})
    return out


def apply(text: str, secs: list[Section], section_key: str, op: str, index: int,
          values: dict) -> tuple[str | None, list[str], int]:
    """Apply one operation (save, duplicate, delete, add) to a draft's text.

    Returns (new_text, errors, focus_index). ``new_text`` is None when nothing was changed."""
    from .custom import CustomError, parse_yaml

    sec = next((s for s in secs if s.key == section_key), None)
    if sec is None:
        return None, ["Unknown section."], 0
    try:
        suite = parse_yaml(text)
    except CustomError as exc:
        return None, [str(exc)], 0

    if sec.single:
        errors = sec.write(suite, values)
        return (None, errors, 0) if errors else (dump(suite), [], 0)

    items = _container(suite, sec.container, create=True)
    if not isinstance(items, list):
        return None, ["This part of the suite is not a list."], 0

    if op == "add":
        if not sec.addable:
            return None, ["Tests cannot be added here."], 0
        items.append(sec.blank(suite))
        return dump(suite), [], len(items) - 1
    if not 0 <= index < len(items):
        return None, ["That test no longer exists."], 0
    if op == "delete":
        if not sec.deletable:
            return None, ["This cannot be deleted."], 0
        del items[index]
        return dump(suite), [], max(0, min(index, len(items) - 1))
    if op == "duplicate":
        if not sec.duplicable:
            return None, ["This cannot be duplicated."], 0
        clone = copy.deepcopy(items[index])
        clone["id"] = next_id(items, re.sub(r"\.\d+$", "", str(clone.get("id", sec.id_prefix))) + ".copy")
        items.insert(index + 1, clone)
        return dump(suite), [], index + 1
    if op == "save":
        errors = sec.write(items[index], values)
        return (None, errors, index) if errors else (dump(suite), [], index)
    return None, ["Unknown action."], index


# Error paths from the validator, mapped to the item they belong to.
_PATH = re.compile(r"^(?P<container>[a-z_.]+)\[(?P<index>\d+)\]")
_CONTAINERS = {"items": "items", "hazards": "hazards", "probes_inline.controls": "controls",
               "probes_inline.injection": "injection"}


def problems_by_item(problems) -> tuple[dict[tuple[str, int], list], list]:
    """Group validator problems under the test they belong to. The rest are suite-wide."""
    grouped: dict[tuple[str, int], list] = {}
    general = []
    for p in problems:
        m = _PATH.match(p.path or "")
        if m and m.group("container") in _CONTAINERS:
            grouped.setdefault((_CONTAINERS[m.group("container")], int(m.group("index"))), []).append(p)
        else:
            general.append(p)
    return grouped, general


def form_values(form: dict, fields: list[Field]) -> dict:
    """Pull the values for these fields out of a submitted form (field names are f_<key>)."""
    out: dict = {}
    for f in fields:
        if f.readonly:
            continue
        name = f"f_{f.key}"
        if f.kind == "check":
            out[f.key] = name in form
        elif f.kind == "multicheck":
            out[f.key] = form.getlist(name) if hasattr(form, "getlist") else list(form.get(name) or [])
        elif name in form:
            out[f.key] = form[name]
    return out
