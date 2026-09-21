"""Model adapters.

Every provider is normalised to one interface so a suite does not know or care
which vendor is behind a target. Vendor-specific behaviour — refusal shapes,
content-filter responses, token accounting — is absorbed here (REQ-M-04, REQ-M-06).
"""

from .base import Completion, ProviderError, PROVIDERS, get_adapter

__all__ = ["Completion", "ProviderError", "PROVIDERS", "get_adapter"]
