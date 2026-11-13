"""Is a model target on the public cloud, or on-prem?

The provider alone is not enough: "On-prem / OpenAI-compatible" accepts any URL, including a
public service such as https://api.fireworks.ai/inference/v1. Restricted content (REQ-M-03,
REQ-NF-03) must follow where the requests actually go, so an OpenAI-compatible endpoint counts
as public cloud when its host is a known cloud AI service, or resolves to a public address and
is not listed in ``AISH_ONPREM_HOSTS``.

A host that does not resolve from this server is treated as on-prem: it is most likely an
internal DNS name, and a request to a name that does not resolve reaches no one. The same
check runs again before every run, when it must resolve to be called at all.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from urllib.parse import urlsplit

from ..config import get_settings
from .base import CLOUD_PROVIDERS

# Domains of public AI APIs. Matching is by suffix on a label boundary.
CLOUD_HOST_SUFFIXES = (
    "fireworks.ai", "openrouter.ai", "openai.com", "anthropic.com", "deepseek.com",
    "together.xyz", "together.ai", "groq.com", "mistral.ai", "googleapis.com", "perplexity.ai",
    "azure.com", "x.ai", "cerebras.ai", "huggingface.co", "hf.space", "cohere.ai", "cohere.com",
    "amazonaws.com", "replicate.com", "deepinfra.com", "novita.ai", "hyperbolic.xyz",
    "sambanova.ai", "nebius.com", "nebius.ai", "lambdalabs.com", "lambda.ai", "featherless.ai",
    "anyscale.com", "octoai.cloud", "baseten.co", "modal.run", "runpod.net", "ai21.com",
    "moonshot.ai", "moonshot.cn", "aliyuncs.com", "bigmodel.cn", "minimax.io", "voyageai.com",
)

# Hosts named by a dedicated provider, to point the user at it.
DEDICATED = {"fireworks.ai": "fireworks", "openrouter.ai": "openrouter", "openai.com": "openai",
             "anthropic.com": "anthropic", "deepseek.com": "deepseek"}

_TTL = 300.0
_lookup = socket.getaddrinfo  # replaced in tests, which must not depend on real DNS
_cache: dict[str, tuple[float, bool]] = {}


def _matches(host: str, suffixes) -> str | None:
    for suffix in suffixes:
        suffix = suffix.strip().lower().lstrip(".")
        if suffix and (host == suffix or host.endswith("." + suffix)):
            return suffix
    return None


def _onprem_hosts() -> list[str]:
    raw = get_settings().onprem_hosts or ""
    return [h for h in (part.strip().lower() for part in raw.split(",")) if h]


def _resolves_public(host: str) -> bool:
    now = time.monotonic()
    hit = _cache.get(host)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        address = ipaddress.ip_address(host)
        addresses = [address]
    except ValueError:
        try:
            infos = _lookup(host, 443)
        except (socket.gaierror, UnicodeError):
            addresses = []
        else:
            addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    public = any(a.is_global for a in addresses)
    _cache[host] = (now, public)
    return public


def host_of(endpoint: str | None) -> str:
    try:
        return (urlsplit(endpoint or "").hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def is_cloud_target(provider: str, endpoint: str | None = None) -> bool:
    """True when requests for this model leave for a public cloud service."""
    if provider in CLOUD_PROVIDERS:
        return True
    host = host_of(endpoint)
    if not host:
        return False
    if _matches(host, CLOUD_HOST_SUFFIXES):
        return True  # a known AI cloud is cloud even if listed as on-prem
    if _matches(host, _onprem_hosts()):
        return False
    return _resolves_public(host)


def dedicated_provider(provider: str, endpoint: str | None) -> str | None:
    """The dedicated provider id to use instead of OpenAI-compatible for this URL, if any."""
    if provider in CLOUD_PROVIDERS:
        return None
    suffix = _matches(host_of(endpoint), DEDICATED)
    return DEDICATED.get(suffix) if suffix else None
