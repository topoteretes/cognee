"""Entra ID (managed identity) authentication for Azure OpenAI.

Shared by both structured-output frameworks: the instructor ``AzureOpenAIAdapter``
hands the token provider to the OpenAI client, and ``litellm_native`` passes it to
litellm as ``azure_ad_token_provider``. Both need the same credential, scope and
endpoint rule, so they live here once.
"""

from collections.abc import Callable
from functools import lru_cache

# The token audience of Azure OpenAI (Azure AI Services).
AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


@lru_cache(maxsize=1)
def get_azure_ad_token_provider() -> Callable[[], str]:
    """The process-wide Entra ID token provider for Azure managed identity.

    Built once: the provider caches its token and refreshes it before expiry, and
    the adapters are rebuilt per call (``litellm_native``) or per configuration, so a
    provider per adapter would fetch a new token from the identity endpoint far
    more often than needed.
    """
    try:
        from azure.identity import (  # ty:ignore[unresolved-import]
            DefaultAzureCredential,
            get_bearer_token_provider,
        )
    except ImportError as error:
        raise ImportError(
            "azure-identity is required for Azure managed identity authentication. "
            'Install it with: pip install "cognee[azure]"'
        ) from error

    return get_bearer_token_provider(DefaultAzureCredential(), AZURE_COGNITIVE_SERVICES_SCOPE)


def require_azure_endpoint(endpoint: str | None) -> str:
    """Return ``endpoint``, or raise: managed identity has no default Azure endpoint."""
    if not endpoint:
        raise ValueError(
            "LLM_ENDPOINT must be set to your Azure OpenAI endpoint "
            "(e.g. https://YOUR-RESOURCE.openai.azure.com) when using Azure managed identity."
        )
    return endpoint
