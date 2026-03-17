from .base import TranslationProvider, TranslationResult
from .llm_provider import LLMTranslationProvider
from .google_provider import GoogleTranslationProvider
from .azure_provider import AzureTranslationProvider

__all__ = [
    "TranslationProvider",
    "TranslationResult",
    "LLMTranslationProvider",
    "GoogleTranslationProvider",
    "AzureTranslationProvider",
    "get_translation_provider",
]


def get_translation_provider(provider_name: str) -> TranslationProvider:
    """
    Factory function to get the appropriate translation provider.

    Args:
        provider_name: Name of the provider:
            - "llm": Uses the configured LLM (OpenAI, Azure, Ollama, Anthropic, etc.)
            - "google": Uses Google Cloud Translation API
            - "azure": Uses Azure Translator API

    Returns:
        TranslationProvider instance

    Raises:
        ValueError: If the provider name is not recognized
    """
    providers = {
        "llm": LLMTranslationProvider,
        "google": GoogleTranslationProvider,
        "azure": AzureTranslationProvider,
    }

    if provider_name.lower() not in providers:
        raise ValueError(
            f"Unknown translation provider: {provider_name}. "
            f"Available providers: {list(providers.keys())}"
        )

    return providers[provider_name.lower()]()
