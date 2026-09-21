"""Cryptography, password handling, CSRF and outbound-request safety.

Every primitive here is from a vetted library. Nothing in this module invents a
scheme: Argon2id for passwords, AES-256-GCM for stored provider credentials,
HMAC-SHA256 for signed values, and `secrets` for every random token.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import re
import secrets
import socket
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from .config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #

_TOKEN_BYTES = 32


def _hasher(settings: Settings) -> PasswordHasher:
    return PasswordHasher(
        time_cost=settings.argon2_time_cost,
        memory_cost=settings.argon2_memory_cost_kib,
        parallelism=settings.argon2_parallelism,
        hash_len=32,
        salt_len=16,
        type=Type.ID,  # Argon2id
    )


def normalise_password(password: str) -> str:
    """NFKC per NIST SP 800-63B so the same typed password always hashes alike."""
    return unicodedata.normalize("NFKC", password)


def hash_password(password: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return _hasher(settings).hash(normalise_password(password))


def verify_password(stored_hash: str, password: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    try:
        return _hasher(settings).verify(stored_hash, normalise_password(password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    try:
        return _hasher(settings).check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True


# A deliberately slow no-op used when an account does not exist, so that the
# response time of a login attempt does not disclose whether the email is known.
_DUMMY_HASH: str | None = None


def dummy_verify(settings: Settings | None = None) -> None:
    global _DUMMY_HASH
    settings = settings or get_settings()
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(24), settings)
    verify_password(_DUMMY_HASH, "not-the-password", settings)


class PasswordPolicyError(ValueError):
    pass


# Blocking the handful of passwords that dominate credential-stuffing lists.
# Not a substitute for length; a supplement to it.
_COMMON_PASSWORDS = {
    "password", "password1", "password123", "123456789", "1234567890",
    "qwertyuiop", "administrator", "letmein123", "welcome1234", "iloveyou123",
    "changeme123", "passw0rd123", "adminadmin", "qwerty123456",
}


def check_password_policy(password: str, email: str = "", settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    password = normalise_password(password)
    if len(password) < settings.password_min_length:
        raise PasswordPolicyError(
            f"Password must be at least {settings.password_min_length} characters."
        )
    if len(password) > settings.password_max_length:
        raise PasswordPolicyError("Password is too long.")
    lowered = password.lower()
    if lowered in _COMMON_PASSWORDS:
        raise PasswordPolicyError("That password appears in breach corpora. Choose another.")
    local_part = email.split("@", 1)[0].lower()
    if local_part and len(local_part) > 3 and local_part in lowered:
        raise PasswordPolicyError("Password must not contain your email address.")
    if len(set(password)) < 5:
        raise PasswordPolicyError("Password is too repetitive.")


# --------------------------------------------------------------------------- #
# Tokens, signing and constant-time comparison
# --------------------------------------------------------------------------- #


def new_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def token_fingerprint(token: str) -> str:
    """Session tokens are stored as a digest, so a database read cannot replay one."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equals(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def sign(value: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    mac = hmac.new(settings.secret_key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).decode("ascii").rstrip("=")


def verify_signature(value: str, signature: str, settings: Settings | None = None) -> bool:
    return constant_time_equals(sign(value, settings), signature)


# --------------------------------------------------------------------------- #
# Provider credential encryption (AES-256-GCM, key derived via HKDF)
# --------------------------------------------------------------------------- #


class DecryptionError(Exception):
    pass


def _aead(settings: Settings) -> AESGCM:
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"aish.provider-credentials.v1",
        info=b"aes-256-gcm",
    ).derive(settings.encryption_key.encode("utf-8"))
    return AESGCM(key)


def encrypt_secret(plaintext: str, aad: str = "", settings: Settings | None = None) -> str:
    """Encrypt a provider API key for storage.

    ``aad`` binds the ciphertext to its owner, so a row copied to another user's
    record fails to decrypt instead of silently working.
    """
    settings = settings or get_settings()
    nonce = secrets.token_bytes(12)
    ciphertext = _aead(settings).encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8"))
    return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")


def decrypt_secret(blob: str, aad: str = "", settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    try:
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        nonce, ciphertext = raw[:12], raw[12:]
        return _aead(settings).decrypt(nonce, ciphertext, aad.encode("utf-8")).decode("utf-8")
    except Exception as exc:  # narrow rethrow: never leak cryptographic detail upward
        raise DecryptionError("stored credential could not be decrypted") from exc


def secret_hint(plaintext: str) -> str:
    """What the UI is allowed to show. A stored key is never rendered in full."""
    tail = plaintext[-4:] if len(plaintext) >= 8 else ""
    return f"****{tail}" if tail else "****"


# --------------------------------------------------------------------------- #
# Outbound request safety (SSRF)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EndpointCheck:
    ok: bool
    reason: str = ""
    # True when the host could not be resolved from here. An internal DNS name may
    # legitimately be unresolvable from the web tier, so configuration time treats
    # this as a warning while request time treats it as a refusal.
    unresolved: bool = False


_ALLOWED_SCHEMES = {"https", "http"}


def validate_endpoint(url: str, settings: Settings | None = None) -> EndpointCheck:
    """Vet a user-supplied model endpoint before the server will call it.

    On-prem EUIBA models legitimately live on private networks, so private ranges
    are permitted when ``allow_private_endpoints`` is set — but loopback, link-local
    and cloud metadata addresses are refused unconditionally, because those are the
    targets that turn a model endpoint field into an SSRF primitive.
    """
    settings = settings or get_settings()
    try:
        parts = urlsplit(url)
    except ValueError:
        return EndpointCheck(False, "endpoint is not a valid URL")

    if parts.scheme not in _ALLOWED_SCHEMES:
        return EndpointCheck(False, "endpoint must use http or https")
    if parts.scheme == "http" and not settings.allow_private_endpoints:
        return EndpointCheck(False, "plain http endpoints are not permitted")
    if not parts.hostname:
        return EndpointCheck(False, "endpoint has no host")
    if parts.username or parts.password:
        return EndpointCheck(False, "credentials must not be embedded in the endpoint URL")

    host = parts.hostname.lower()
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".localhost"):
        return EndpointCheck(False, "endpoint host is not permitted")

    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        return EndpointCheck(False, "endpoint host does not resolve from this server",
                             unresolved=True)

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_loopback or address.is_link_local or address.is_multicast or address.is_reserved:
            return EndpointCheck(False, "endpoint resolves to a blocked address range")
        if address in ipaddress.ip_network("169.254.169.254/32"):
            return EndpointCheck(False, "endpoint resolves to a cloud metadata address")
        if address.is_private and not settings.allow_private_endpoints:
            return EndpointCheck(False, "endpoint resolves to a private address range")
    return EndpointCheck(True)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def normalise_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email)) and len(email) <= 254


def redact(text: str, *secrets_to_hide: str) -> str:
    """Strip known secrets out of anything destined for a log or an error page."""
    for secret in secrets_to_hide:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "****")
    return text
