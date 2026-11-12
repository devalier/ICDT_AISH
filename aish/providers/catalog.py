"""Model discovery: ask the provider which models the user's own key can use.

Bring your own key: the key the user pastes into the form is used for this one call and is
never stored. Each entry is an id (what goes in the model field) and a short human label
(name, context length, price where the provider publishes one).
"""

from __future__ import annotations

import httpx

from ..security import validate_endpoint
from .base import ProviderError

_DEFAULT_BASES = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "anthropic": "https://api.anthropic.com/v1",
}
_CHAT_PATH = "/chat/completions"
_MAX_BYTES = 8_000_000            # a model list larger than this is not a model list
FIREWORKS_MODELS = "https://api.fireworks.ai/v1/accounts/fireworks/models"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

# Tests replace this with an httpx.MockTransport; None means the real network.
_TRANSPORT: httpx.AsyncBaseTransport | None = None


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=_TRANSPORT)


def _check(response: httpx.Response) -> dict:
    if response.status_code in (401, 403):
        raise ProviderError("provider rejected the API key", status=response.status_code)
    if response.status_code >= 400:
        raise ProviderError(f"provider returned {response.status_code}", status=response.status_code)
    if len(response.content) > _MAX_BYTES:
        raise ProviderError("provider returned an oversized model list")
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError("provider returned an unrecognised model list") from exc
    if not isinstance(body, dict):
        raise ProviderError("provider returned an unrecognised model list")
    return body


def _context(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    return f"{n // 1000}k context" if n >= 1000 else f"{n} context"


def _per_million(value) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


async def _openrouter(api_key: str | None, timeout: float) -> list[dict]:
    """OpenRouter publishes its catalogue; a key is sent if given but not needed to list."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with _client(timeout) as client:
        body = _check(await client.get(OPENROUTER_MODELS, headers=headers))
    out = []
    for m in body.get("data") or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        arch = m.get("architecture") or {}
        outputs = arch.get("output_modalities") or []
        if outputs and "text" not in outputs:
            continue                    # image, audio or video generators cannot answer a prompt
        endpoints = m.get("endpoints")
        if isinstance(endpoints, list) and endpoints and not any(
                str(e).endswith("/chat/completions") for e in endpoints):
            continue                    # not reachable through the chat API the harness uses
        pricing = m.get("pricing") or {}
        p_in, p_out = _per_million(pricing.get("prompt")), _per_million(pricing.get("completion"))
        parts = [str(m.get("name") or m["id"])]
        if m.get("context_length"):
            parts.append(_context(m["context_length"]))
        if p_in == 0 and p_out == 0:
            parts.append("free")
        elif p_in is not None and p_out is not None:
            parts.append(f"${p_in:.2f} in / ${p_out:.2f} out per 1M tokens")
        if (m.get("reasoning") or {}).get("mandatory"):
            parts.append("reasoning")
        out.append({"id": str(m["id"]), "label": " · ".join(parts)})
    return out


async def _fireworks(api_key: str | None, timeout: float) -> list[dict]:
    """Fireworks' serverless catalogue (the account "fireworks"), as the user's key sees it.
    Models in the user's own account or on a dedicated deployment can still be typed in."""
    if not api_key:
        raise ProviderError("an API key is required to list models")
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    params: dict = {"pageSize": 200, "filter": "supports_serverless=true"}
    models: list[dict] = []
    async with _client(timeout) as client:
        for _page in range(10):                         # at most 2,000 models
            response = await client.get(FIREWORKS_MODELS, headers=headers, params=params)
            if response.status_code == 400 and "filter" in params:
                params.pop("filter")                    # filter syntax not accepted: filter here instead
                response = await client.get(FIREWORKS_MODELS, headers=headers, params=params)
            body = _check(response)
            models.extend(m for m in body.get("models") or [] if isinstance(m, dict))
            token = body.get("nextPageToken")
            if not token:
                break
            params["pageToken"] = token
    out = []
    for m in models:
        name = str(m.get("name") or "")
        if not name:
            continue
        if m.get("supportsServerless") is False:
            continue                    # needs a dedicated deployment: not callable with just a key
        if str(m.get("kind") or "") in ("EMBEDDING_MODEL", "DRAFT_ADDON", "FLUMINA_BASE_MODEL", "FLUMINA_ADDON"):
            continue                    # not chat models
        if str(m.get("state") or "READY") not in ("READY", "STATE_UNSPECIFIED"):
            continue
        parts = [str(m.get("displayName") or name.rsplit("/", 1)[-1])]
        if m.get("contextLength"):
            parts.append(_context(m["contextLength"]))
        if m.get("supportsImageInput"):
            parts.append("vision")
        if m.get("deprecationDate"):
            d = m["deprecationDate"]
            when = f"{d.get('year')}-{d.get('month'):02d}-{d.get('day'):02d}" if isinstance(d, dict) and d.get("year") else str(d)
            parts.append(f"retiring {when}")
        out.append({"id": name, "label": " · ".join(parts)})
    return out


async def list_model_entries(
    provider: str, api_key: str | None, endpoint: str | None, timeout: float = 20.0
) -> list[dict]:
    """[{id, label}], sorted by id."""
    try:
        if provider == "openrouter":
            entries = await _openrouter(api_key, timeout)
        elif provider == "fireworks":
            entries = await _fireworks(api_key, timeout)
        else:
            entries = [{"id": i, "label": ""} for i in await list_models(provider, api_key, endpoint, timeout)]
    except httpx.RequestError as exc:
        raise ProviderError(f"could not reach provider: {type(exc).__name__}", retryable=True) from exc
    unique = {e["id"]: e for e in entries}
    return [unique[k] for k in sorted(unique)]


async def list_models(
    provider: str, api_key: str | None, endpoint: str | None, timeout: float = 15.0
) -> list[str]:
    if provider in ("openrouter", "fireworks"):
        return [e["id"] for e in await list_model_entries(provider, api_key, endpoint, timeout)]
    if provider not in _DEFAULT_BASES and provider != "openai_compatible":
        raise ProviderError(f"unknown provider {provider!r}")

    base = (endpoint or _DEFAULT_BASES.get(provider, "")).rstrip("/")
    if not base:
        raise ProviderError("an endpoint URL is required to list models")
    if endpoint or provider == "openai_compatible":
        check = validate_endpoint(base)
        if not check.ok:
            raise ProviderError(f"endpoint rejected: {check.reason}")
    if base.endswith(_CHAT_PATH):
        base = base[: -len(_CHAT_PATH)]

    headers = {"Accept": "application/json"}
    if provider == "anthropic":
        if not api_key:
            raise ProviderError("an API key is required to list models")
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        params = {"limit": 1000}
    else:
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider != "openai_compatible":
            raise ProviderError("an API key is required to list models")
        params = {}

    try:
        async with _client(timeout) as client:
            response = await client.get(f"{base}/models", headers=headers, params=params)
    except httpx.RequestError as exc:
        raise ProviderError(f"could not reach provider: {type(exc).__name__}", retryable=True) from exc

    body = _check(response)
    try:
        ids = sorted({str(m["id"]) for m in body["data"]})
    except (KeyError, TypeError) as exc:
        raise ProviderError("provider returned an unrecognised model list") from exc
    return ids
