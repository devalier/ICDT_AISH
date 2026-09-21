"""Anthropic adapter, via the official SDK."""

from __future__ import annotations

import time

import anthropic

from .base import Completion, GenerationParams, ProviderError

# Models on which sampling parameters and budget_tokens were removed from the API.
# Sending temperature/top_p to one of these returns a 400, so the adapter drops them
# and records that the run was not temperature-pinned.
_NO_SAMPLING_PREFIXES = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                         "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")


def _supports_sampling(model: str) -> bool:
    return not model.startswith(_NO_SAMPLING_PREFIXES)


class AnthropicAdapter:
    provider_id = "anthropic"
    requires_endpoint = False
    requires_key = True

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        api_key: str | None,
        endpoint: str | None,
        params: GenerationParams,
        timeout: float,
    ) -> Completion:
        if not api_key:
            raise ProviderError("an Anthropic API key is required for this target")

        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)
        request: dict = {
            "model": model,
            "max_tokens": params.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        if _supports_sampling(model):
            request["temperature"] = params.temperature
            request["top_p"] = params.top_p

        started = time.perf_counter()
        try:
            response = await client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise ProviderError("authentication rejected by Anthropic", status=401) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderError("API key lacks permission for this model", status=403) from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"model {model!r} not found", status=404) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("rate limited by Anthropic", retryable=True, status=429) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic returned {exc.status_code}", retryable=exc.status_code >= 500,
                status=exc.status_code,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("could not reach Anthropic", retryable=True) from exc
        finally:
            await client.close()

        latency_ms = int((time.perf_counter() - started) * 1000)
        text = "".join(block.text for block in response.content if block.type == "text")
        stop_reason = response.stop_reason or ""
        return Completion(
            text=text,
            tokens_in=response.usage.input_tokens or 0,
            tokens_out=response.usage.output_tokens or 0,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            refused=stop_reason == "refusal",
        )
