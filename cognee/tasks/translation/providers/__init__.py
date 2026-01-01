from .base import TranslationProvider, TranslationResult
from .openai_provider import OpenAITranslationProvider
from .google_provider import GoogleTranslationProvider
from .azure_provider import AzureTranslationProvider

__all__ = [
    "TranslationProvider",
    "TranslationResult",
    "OpenAITranslationProvider",
    "GoogleTranslationProvider",
    "AzureTranslationProvider",
]


def get_translation_provider(provider_name: str) -> TranslationProvider:
    """
    Factory function to get the appropriate translation provider.

    Args:
        provider_name: Name of the provider ("openai", "google", or "azure")

    Returns:
        TranslationProvider instance

    Raises:
        ValueError: If the provider name is not recognized
    """
    providers = {
        "openai": OpenAITranslationProvider,
        "google": GoogleTranslationProvider,
        "azure": AzureTranslationProvider,
    }

    if provider_name.lower() not in providers:
        raise ValueError(
            f"Unknown translation provider: {provider_name}. "
            f"Available providers: {list(providers.keys())}"
        )

    return providers[provider_name.lower()]()
