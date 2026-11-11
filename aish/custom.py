"""A user's own version of a test suite: loading, validating and fingerprinting it.

The Working Group's suites are YAML. A user may keep a personal copy of any of the four and
change it; this module is what stops that being a way to run code or to reach outside the
suite's own sandbox. Everything is declarative (YAML that is parsed, never executed), size
limited, and checked by the same validator as the official files, plus a few extra rules that
only apply to a user's copy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from harness.loader import Packs, packs_dir
from harness.validate import Problem, validate

from .models import CustomDraft, CustomSuite, utcnow

MAX_BYTES = 200_000


class CustomError(Exception):
    """The text could not be used as a suite. The message is safe to show the user."""


def parse_yaml(text: str) -> dict:
    """Parse untrusted YAML. Anchors and aliases are refused: they let a small document expand
    into a huge one. Only plain mappings, lists and scalars are accepted."""
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise CustomError(f"The file is larger than {MAX_BYTES // 1000} KB.")
    try:
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, (yaml.AliasEvent,)) or getattr(event, "anchor", None):
                raise CustomError("YAML anchors and aliases (& and *) are not allowed.")
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise CustomError(f"This is not valid YAML{where}: {getattr(exc, 'problem', exc)}.") from exc
    if not isinstance(data, dict):
        raise CustomError("A suite must be a YAML mapping (key: value pairs) at the top level.")
    return data


def fingerprint(suite: dict) -> str:
    """Short hash of what the suite means: comments, key order and whitespace do not change it."""
    clean = {k: v for k, v in suite.items() if not str(k).startswith("_")}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def official_text(base: dict) -> str:
    """The official suite file exactly as written (comments included), as a starting point.
    For the safety suite the EUIBA probes are brought inline so they can be edited too."""
    path = packs_dir() / base["_path"]
    text = path.read_text(encoding="utf-8")
    probes_ref = base.get("probes_file")
    if probes_ref:
        probes_text = (packs_dir() / probes_ref).read_text(encoding="utf-8")
        indented = "\n".join(("  " + line) if line.strip() else line for line in probes_text.splitlines())
        text = re.sub(r"^probes_file:.*$", "# The EUIBA probes, brought inline so you can edit them.\n"
                      "probes_inline:\n" + indented.replace("\\", "\\\\"), text, count=1, flags=re.M)
    return text


def _packs_with(view_packs: Packs, key: str, suite: dict) -> Packs:
    tmp = Packs(root=view_packs.root, models=view_packs.models, attributes=view_packs.attributes,
                criteria=view_packs.criteria)
    tmp.suites = {key: suite}
    return tmp


def check(base: dict, text: str, view) -> tuple[dict | None, list[Problem]]:
    """Validate a user's text as their version of ``base``. Returns (suite, problems); the
    suite is only returned when there are no errors."""
    key = "your-version.yaml"
    try:
        suite = parse_yaml(text)
    except CustomError as exc:
        return None, [Problem("error", key, "", str(exc))]

    problems = [p for p in validate(_packs_with(view.packs, key, suite)) if p.file == key]

    def err(path: str, message: str) -> None:
        problems.append(Problem("error", key, path, message))

    if suite.get("id") != base["id"]:
        err("id", f"must stay {base['id']!r}: this is your version of that suite")
    if suite.get("family") != base["family"]:
        err("family", f"must stay {base['family']!r}")
    if suite.get("scorer") != base.get("scorer"):
        err("scorer", f"must stay {base.get('scorer')!r}; scorers are built in and cannot be swapped")
    if suite.get("criteria") != base.get("criteria"):
        err("criteria", f"must stay {base.get('criteria')!r}: thresholds are set by the Working Group")
    if base.get("restricted") and not suite.get("restricted"):
        err("restricted", "this suite holds hazardous content and must stay restricted: true")
    if (suite.get("dataset") or None) != (base.get("dataset") or None):
        err("dataset", "the dataset source and revision cannot be changed in your version")
    if suite.get("probes_file") and suite.get("probes_file") != base.get("probes_file"):
        err("probes_file", "cannot point at another file; use probes_inline instead")
    if suite.get("judge") and suite.get("judge") != base.get("judge"):
        err("judge", "the judge is set by the deployment and cannot be changed here")
    for tag in ("applies_to",):
        if suite.get(tag) != base.get(tag):
            suite[tag] = copy.deepcopy(base.get(tag))      # not user-editable; keep the official value

    errors = [p for p in problems if p.severity == "error"]
    return (None if errors else suite), problems


def active(db: Session, user_id: int, base_suite_id: str) -> CustomSuite | None:
    return db.scalar(
        select(CustomSuite)
        .where(CustomSuite.user_id == user_id, CustomSuite.base_suite_id == base_suite_id,
               CustomSuite.archived_at.is_(None))
        .order_by(CustomSuite.id.desc())
    )


def save(db: Session, user_id: int, base: dict, text: str, suite: dict, note: str = "") -> CustomSuite:
    row = CustomSuite(user_id=user_id, base_suite_id=base["id"], yaml_text=text,
                      fingerprint=fingerprint(suite), note=note[:200])
    db.add(row)
    db.flush()
    return row


def reset(db: Session, user_id: int, base_suite_id: str) -> int:
    """Go back to the official version. Saved versions stay, archived, so past runs still resolve."""
    rows = list(db.scalars(select(CustomSuite).where(
        CustomSuite.user_id == user_id, CustomSuite.base_suite_id == base_suite_id,
        CustomSuite.archived_at.is_(None))))
    for row in rows:
        row.archived_at = utcnow()
    return len(rows)


def load_suite(row: CustomSuite, base: dict) -> dict:
    """The suite dict a stored version stands for, as the runner expects it."""
    suite = parse_yaml(row.yaml_text)
    suite["applies_to"] = copy.deepcopy(base.get("applies_to"))
    suite["_path"] = f"your version ({row.fingerprint})"
    suite["_custom"] = {"id": row.id, "fingerprint": row.fingerprint}
    return suite


def label(suite_version: str, fp: str) -> str:
    return f"{suite_version}+custom.{fp[:6]}"


# ---- drafts ------------------------------------------------------------------------------ #


def get_draft(db: Session, user_id: int, base_suite_id: str) -> CustomDraft | None:
    return db.scalar(select(CustomDraft).where(
        CustomDraft.user_id == user_id, CustomDraft.base_suite_id == base_suite_id))


def put_draft(db: Session, user_id: int, base_suite_id: str, text: str) -> CustomDraft:
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise CustomError(f"The suite is larger than {MAX_BYTES // 1000} KB.")
    draft = get_draft(db, user_id, base_suite_id)
    if draft is None:
        draft = CustomDraft(user_id=user_id, base_suite_id=base_suite_id, yaml_text=text)
        db.add(draft)
    else:
        draft.yaml_text = text
        draft.updated_at = utcnow()
    db.flush()
    return draft


def discard_draft(db: Session, user_id: int, base_suite_id: str) -> None:
    draft = get_draft(db, user_id, base_suite_id)
    if draft is not None:
        db.delete(draft)


def working_text(db: Session, user_id: int, base: dict) -> str:
    """What the editor shows: the draft if there is one, else the user's saved version, else the
    official suite."""
    draft = get_draft(db, user_id, base["id"])
    if draft is not None:
        return draft.yaml_text
    saved = active(db, user_id, base["id"])
    return saved.yaml_text if saved is not None else official_text(base)


def has_unsaved_changes(db: Session, user_id: int, base: dict) -> bool:
    """True if the draft means something different from what is saved (or official)."""
    draft = get_draft(db, user_id, base["id"])
    if draft is None:
        return False
    saved = active(db, user_id, base["id"])
    reference = saved.yaml_text if saved is not None else official_text(base)
    try:
        return fingerprint(parse_yaml(draft.yaml_text)) != fingerprint(parse_yaml(reference))
    except CustomError:
        return True
