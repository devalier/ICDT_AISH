"""Per-user permissions for the restricted (Safety) suite.

Restricted suites hold hazardous prompts. By default they run only on on-prem models (REQ-M-03).
A user may switch on "Allow Safety Harness on Public Cloud Models" for themselves, if the
deployment permits the switch; each run that then uses a cloud model also needs an explicit
acknowledgement. Customising a restricted suite needs a one-time consent to its content notice.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from .config import get_settings
from .models import UserPreference, utcnow
from .providers.base import CLOUD_PROVIDERS


def get(db: Session, user_id: int) -> UserPreference:
    pref = db.get(UserPreference, user_id)
    return pref if pref is not None else UserPreference(user_id=user_id, safety_cloud_allowed=False)


def toggle_available() -> bool:
    return bool(get_settings().allow_safety_on_cloud_toggle)


def cloud_allowed(db: Session, user_id: int) -> bool:
    """True only if the deployment permits the switch and this user has turned it on."""
    return toggle_available() and bool(get(db, user_id).safety_cloud_allowed)


def set_cloud_allowed(db: Session, user_id: int, allowed: bool) -> UserPreference:
    pref = db.get(UserPreference, user_id)
    if pref is None:
        pref = UserPreference(user_id=user_id)
        db.add(pref)
    pref.safety_cloud_allowed = bool(allowed) and toggle_available()
    pref.safety_cloud_changed_at = utcnow()
    db.flush()
    return pref


def content_acknowledged(db: Session, user_id: int) -> bool:
    return get(db, user_id).safety_content_ack_at is not None


def acknowledge_content(db: Session, user_id: int) -> None:
    pref = db.get(UserPreference, user_id)
    if pref is None:
        pref = UserPreference(user_id=user_id)
        db.add(pref)
    pref.safety_content_ack_at = utcnow()
    db.flush()


def is_cloud(provider: str) -> bool:
    return provider in CLOUD_PROVIDERS


SWITCH_NAME = "Allow Safety Harness on Public Cloud Models"
WARNING = ("This harness tests AI model safety by prompting the model with safety-challenging topics, "
           "which may result in API revocation or account suspension by the model provider. Use the "
           "safety harness only if you are certain about its purpose and function.")
