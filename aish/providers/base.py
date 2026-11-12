from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    """One normalised model response."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    stop_reason: str = ""
    # A provider-side refusal or content-filter block is its own outcome, never an
    # empty completion and never a scoring error (REQ-M-06).
    refused: bool = False
    raw: dict = field(default_factory=dict)


class ProviderError(Exception):
    """Transport, auth or protocol failure. Distinguished in results from a
    model refusal and from a scoring error."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int | None = None


class Adapter(Protocol):
    provider_id: str
    requires_endpoint: bool
    requires_key: bool

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
    ) -> Completion: ...


# Every provider the harness can call. ``cloud`` is what keeps restricted (hazardous) content
# off third-party APIs (REQ-M-03): a provider marked cloud can never run a restricted suite or
# judge one. ``endpoint`` says whether the form asks for a URL ("required", "optional" or
# "none": the provider's own fixed API, so a key is never sent anywhere else). ``catalog`` says
# whether the model list can be browsed without a key ("public") or needs one ("key").
PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "requires_endpoint": False,
        "requires_key": True,
        "cloud": True,
        "endpoint": "none",
        "catalog": "key",
        "hint": "Cloud. Model id such as claude-opus-5.",
    },
    "openai": {
        "label": "OpenAI",
        "requires_endpoint": False,
        "requires_key": True,
        "cloud": True,
        "endpoint": "optional",
        "catalog": "key",
        "hint": "Cloud. Model id such as gpt-5.1.",
    },
    "deepseek": {
        "label": "DeepSeek",
        "requires_endpoint": False,
        "requires_key": True,
        "cloud": True,
        "endpoint": "optional",
        "catalog": "key",
        "hint": "Cloud. Model id such as deepseek-chat or deepseek-reasoner.",
    },
    "fireworks": {
        "label": "Fireworks AI",
        "requires_endpoint": False,
        "requires_key": True,
        "cloud": True,
        "endpoint": "none",
        "catalog": "key",
        "hint": "Cloud. Any serverless model (browse the list), such as "
                "accounts/fireworks/models/llama-v3p3-70b-instruct, or the id of a model "
                "or deployment in your own Fireworks account.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "requires_endpoint": False,
        "requires_key": True,
        "cloud": True,
        "endpoint": "none",
        "catalog": "public",
        "hint": "Cloud gateway to hundreds of models from many providers, such as "
                "meta-llama/llama-3.3-70b-instruct. The list can be browsed before you enter a key. "
                "Requests go only to providers that do not store or train on prompts.",
    },
    "openai_compatible": {
        "label": "On-prem / OpenAI-compatible endpoint",
        "requires_endpoint": True,
        "requires_key": False,
        "cloud": False,
        "endpoint": "required",
        "catalog": "key",
        "hint": "vLLM, TGI, Ollama, llama.cpp — anything serving /v1/chat/completions.",
    },
}

CLOUD_PROVIDERS = frozenset(p for p, spec in PROVIDERS.items() if spec["cloud"])


def get_adapter(provider_id: str) -> Adapter:
    from .anthropic_adapter import AnthropicAdapter
    from .openai_adapter import (DeepSeekAdapter, FireworksAdapter, OpenAIAdapter,
                                 OpenAICompatibleAdapter, OpenRouterAdapter)

    adapters: dict[str, Adapter] = {
        "anthropic": AnthropicAdapter(),
        "openai": OpenAIAdapter(),
        "deepseek": DeepSeekAdapter(),
        "fireworks": FireworksAdapter(),
        "openrouter": OpenRouterAdapter(),
        "openai_compatible": OpenAICompatibleAdapter(),
    }
    if provider_id not in adapters:
        raise ProviderError(f"unknown provider {provider_id!r}")
    return adapters[provider_id]
