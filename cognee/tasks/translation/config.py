from functools import lru_cache
from typing import Literal, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


TranslationProviderType = Literal["llm", "google", "azure"]


class TranslationConfig(BaseSettings):
    """
    Configuration settings for the translation task.

    Environment variables can be used to configure these settings:
    - TRANSLATION_PROVIDER: The translation service to use ("llm", "google", "azure")
    - TARGET_LANGUAGE: Default target language (ISO 639-1 code, e.g., "en", "es", "fr")
    - CONFIDENCE_THRESHOLD: Minimum confidence for language detection (0.0 to 1.0)
    - GOOGLE_TRANSLATE_API_KEY: API key for Google Translate
    - GOOGLE_PROJECT_ID: Google Cloud project ID
    - AZURE_TRANSLATOR_KEY: API key for Azure Translator
    - AZURE_TRANSLATOR_REGION: Region for Azure Translator
    - AZURE_TRANSLATOR_ENDPOINT: Endpoint URL for Azure Translator
    - TRANSLATION_BATCH_SIZE: Number of texts to translate per batch
    - TRANSLATION_MAX_RETRIES: Maximum retry attempts on failure
    - TRANSLATION_TIMEOUT_SECONDS: Request timeout in seconds
    """

    # Translation provider settings
    translation_provider: TranslationProviderType = Field(
        default="llm",
        validation_alias=AliasChoices("TRANSLATION_PROVIDER", "translation_provider"),
    )
    target_language: str = Field(
        default="en",
        validation_alias=AliasChoices("TARGET_LANGUAGE", "target_language"),
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("CONFIDENCE_THRESHOLD", "confidence_threshold"),
    )

    # Google Translate settings
    google_translate_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_TRANSLATE_API_KEY", "google_translate_api_key"),
    )
    google_project_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_PROJECT_ID", "google_project_id"),
    )

    # Azure Translator settings
    azure_translator_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AZURE_TRANSLATOR_KEY", "azure_translator_key"),
    )
    azure_translator_region: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AZURE_TRANSLATOR_REGION", "azure_translator_region"),
    )
    azure_translator_endpoint: str = Field(
        default="https://api.cognitive.microsofttranslator.com",
        validation_alias=AliasChoices("AZURE_TRANSLATOR_ENDPOINT", "azure_translator_endpoint"),
    )

    # LLM provider uses the existing LLM configuration

    # Performance settings (with TRANSLATION_ prefix for env vars)
    batch_size: int = Field(
        default=10,
        validation_alias=AliasChoices("TRANSLATION_BATCH_SIZE", "batch_size"),
    )
    max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("TRANSLATION_MAX_RETRIES", "max_retries"),
    )
    timeout_seconds: int = Field(
        default=30,
        validation_alias=AliasChoices("TRANSLATION_TIMEOUT_SECONDS", "timeout_seconds"),
    )

    # Language detection settings
    min_text_length_for_detection: int = 10
    skip_detection_for_short_text: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "translation_provider": self.translation_provider,
            "target_language": self.target_language,
            "confidence_threshold": self.confidence_threshold,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }


@lru_cache()
def get_translation_config() -> TranslationConfig:
    """Get the translation configuration singleton."""
    return TranslationConfig()


def clear_translation_config_cache():
    """Clear the cached config for testing purposes."""
    get_translation_config.cache_clear()
