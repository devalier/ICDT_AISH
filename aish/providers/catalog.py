"""Model discovery: ask the provider which model identifiers a key can use."""

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


async def list_models(
    provider: str, api_key: str | None, endpoint: str | None, timeout: float = 15.0
) -> list[str]:
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
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.get(f"{base}/models", headers=headers, params=params)
    except httpx.RequestError as exc:
        raise ProviderError(f"could not reach provider: {type(exc).__name__}", retryable=True) from exc

    if response.status_code in (401, 403):
        raise ProviderError("provider rejected the API key", status=response.status_code)
    if response.status_code >= 400:
        raise ProviderError(f"provider returned {response.status_code}", status=response.status_code)
    try:
        ids = sorted({str(m["id"]) for m in response.json()["data"]})
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderError("provider returned an unrecognised model list") from exc
    return ids
