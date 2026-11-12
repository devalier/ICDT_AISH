"""OpenAI and OpenAI-compatible adapters.

Raw HTTP via httpx rather than a vendor SDK: the wire format is one stable
endpoint, and every on-prem server in the EUIBA fleet (vLLM, TGI, Ollama,
llama.cpp) speaks it. One fewer dependency is one fewer supply-chain surface.
"""

from __future__ import annotations

import re
import time

import httpx

from ..config import get_settings
from ..security import validate_endpoint
from .base import Completion, GenerationParams, ProviderError

_CHAT_PATH = "/chat/completions"
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Tests replace this with an httpx.MockTransport; None means the real network.
_TRANSPORT: httpx.AsyncBaseTransport | None = None

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Open reasoning models served over the chat API (DeepSeek-R1, Qwen and others on vLLM,
    Fireworks, OpenRouter) can put their chain of thought in the reply inside <think> tags.
    Scoring must read the answer, not the thinking, or a letter mentioned while reasoning is
    taken as the answer. An unterminated block (the reply was cut off mid-thought) leaves no
    answer at all."""
    if text.lstrip().startswith("<think>") and "</think>" not in text:
        return ""
    return _THINK.sub("", text, count=1)


def _error_detail(response: httpx.Response) -> str:
    """The provider's own error message, so a 400/403 says what was actually wrong."""
    try:
        err = response.json().get("error")
        text = err.get("message") if isinstance(err, dict) else err
    except (ValueError, AttributeError):
        text = None
    return str(text or "")[:300]


async def _post_chat(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    system: str | None,
    params: GenerationParams,
    timeout: float,
    verify_endpoint: bool,
    limit_param: str = "max_tokens",
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
    moderation_is_refusal: bool = False,
) -> Completion:
    if verify_endpoint:
        check = validate_endpoint(base_url)
        if not check.ok:
            raise ProviderError(f"endpoint rejected: {check.reason}")

    url = base_url.rstrip("/")
    if not url.endswith(_CHAT_PATH):
        url = url + _CHAT_PATH

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": params.temperature,
        "top_p": params.top_p,
        limit_param: params.max_tokens,
        "stream": False,
    }
    if params.seed is not None:
        payload["seed"] = params.seed
    payload.update(extra_body or {})

    headers = {"Content-Type": "application/json", "Accept": "application/json", **(extra_headers or {})}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.perf_counter()
    try:
        # follow_redirects stays off: a redirect is how a validated endpoint turns
        # into an unvalidated one.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False,
                                     transport=_TRANSPORT) as client:
            response = await client.post(url, json=payload, headers=headers)
            # Reasoning models reject fixed sampling parameters. Retry once without
            # whichever ones the endpoint names; the run records that it was not pinned.
            if response.status_code == 400:
                detail = _error_detail(response).lower()
                dropped = [k for k in ("temperature", "top_p", "seed") if k in detail and k in payload]
                if dropped:
                    for key in dropped:
                        payload.pop(key)
                    response = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise ProviderError("model endpoint timed out", retryable=True) from exc
    except httpx.RequestError as exc:
        raise ProviderError(f"could not reach model endpoint: {type(exc).__name__}", retryable=True) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code == 401:
        raise ProviderError("authentication rejected by the model endpoint", status=401)
    if response.status_code == 402:
        raise ProviderError(
            f"the provider account has insufficient credit: {_error_detail(response) or 'top up and retry'}",
            status=402,
        )
    if response.status_code == 403 and moderation_is_refusal:
        detail = _error_detail(response).lower()
        if "flagged" in detail or "moderation" in detail:
            # The gateway's own moderation blocked the prompt before any model saw it. For the
            # harness that is a refusal by the system under test, not a transport failure.
            return Completion(text="", latency_ms=int((time.perf_counter() - started) * 1000),
                              stop_reason="moderation", refused=True)
    if response.status_code == 403:
        raise ProviderError(
            f"forbidden by the provider: {_error_detail(response) or 'API key lacks permission for this model'}",
            status=403,
        )
    if response.status_code == 404:
        raise ProviderError(f"model {model!r} or endpoint path not found", status=404)
    if response.status_code == 429:
        raise ProviderError("rate limited by the model endpoint", retryable=True, status=429)
    if response.status_code >= 500:
        raise ProviderError(
            f"model endpoint returned {response.status_code}", retryable=True,
            status=response.status_code,
        )
    if response.status_code >= 400:
        raise ProviderError(
            f"model endpoint rejected the request ({response.status_code}): "
            f"{_error_detail(response) or 'no detail'}",
            status=response.status_code,
        )

    try:
        body = response.json()
        if isinstance(body, dict) and body.get("error") and not body.get("choices"):
            raise ProviderError(f"model endpoint reported an error: {_error_detail(response) or 'no detail'}")
        choice = body["choices"][0]
        if choice.get("error"):
            # A gateway can return 200 and report an upstream failure on the choice itself.
            err = choice["error"]
            raise ProviderError(f"upstream model error: {err.get('message') if isinstance(err, dict) else err}"[:300],
                                retryable=True)
        text = _strip_reasoning(choice["message"].get("content") or "")
        finish = choice.get("finish_reason") or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError("model endpoint returned an unrecognised response shape") from exc

    usage = body.get("usage") or {}
    return Completion(
        text=text,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        latency_ms=latency_ms,
        stop_reason=finish,
        refused=finish == "content_filter",
    )


class OpenAIAdapter:
    provider_id = "openai"
    requires_endpoint = False
    requires_key = True

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout) -> Completion:
        if not api_key:
            raise ProviderError("an OpenAI API key is required for this model")
        return await _post_chat(
            base_url=endpoint or "https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            prompt=prompt,
            system=system,
            params=params,
            timeout=timeout,
            verify_endpoint=bool(endpoint),
            limit_param="max_completion_tokens",
        )


class DeepSeekAdapter:
    provider_id = "deepseek"
    requires_endpoint = False
    requires_key = True

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout) -> Completion:
        if not api_key:
            raise ProviderError("a DeepSeek API key is required for this model")
        return await _post_chat(
            base_url=endpoint or "https://api.deepseek.com",
            api_key=api_key,
            model=model,
            prompt=prompt,
            system=system,
            params=params,
            timeout=timeout,
            verify_endpoint=bool(endpoint),
        )


class FireworksAdapter:
    """Fireworks AI: OpenAI-compatible chat completions on Fireworks' own fixed API. Any model
    id Fireworks accepts works: a serverless model (accounts/fireworks/models/...), a model in
    the user's own account, or a dedicated deployment."""

    provider_id = "fireworks"
    requires_endpoint = False
    requires_key = True

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout) -> Completion:
        if not api_key:
            raise ProviderError("a Fireworks API key is required for this model")
        return await _post_chat(
            base_url=FIREWORKS_BASE,           # fixed: the key is never sent anywhere else
            api_key=api_key,
            model=model,
            prompt=prompt,
            system=system,
            params=params,
            timeout=timeout,
            verify_endpoint=False,
        )


class OpenRouterAdapter:
    """OpenRouter: one OpenAI-compatible API in front of many upstream providers."""

    provider_id = "openrouter"
    requires_endpoint = False
    requires_key = True

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout) -> Completion:
        if not api_key:
            raise ProviderError("an OpenRouter API key is required for this model")
        settings = get_settings()
        policy = "allow" if settings.openrouter_data_collection.lower() == "allow" else "deny"
        return await _post_chat(
            base_url=OPENROUTER_BASE,          # fixed: the key is never sent anywhere else
            api_key=api_key,
            model=model,
            prompt=prompt,
            system=system,
            params=params,
            timeout=timeout,
            verify_endpoint=False,
            # Attribution headers OpenRouter documents (optional, identify the app, carry no data).
            extra_headers={"HTTP-Referer": f"https://{settings.public_host}",
                           "X-Title": settings.site_name},
            # Route only to upstream providers that do not store or train on prompts, unless the
            # deployment explicitly allows it (AISH_OPENROUTER_DATA_COLLECTION=allow).
            extra_body={"provider": {"data_collection": policy}},
            moderation_is_refusal=True,
        )


class OpenAICompatibleAdapter:
    provider_id = "openai_compatible"
    requires_endpoint = True
    requires_key = False

    async def generate(self, *, model, prompt, system, api_key, endpoint, params, timeout) -> Completion:
        if not endpoint:
            raise ProviderError("an endpoint URL is required for an on-prem model")
        _ = get_settings()
        return await _post_chat(
            base_url=endpoint,
            api_key=api_key,
            model=model,
            prompt=prompt,
            system=system,
            params=params,
            timeout=timeout,
            verify_endpoint=True,
        )
