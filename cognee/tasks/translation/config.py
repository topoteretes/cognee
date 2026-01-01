from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


TranslationProviderType = Literal["openai", "google", "azure"]


class TranslationConfig(BaseSettings):
    """
    Configuration settings for the translation task.

    Environment variables can be used to configure these settings:
    - TRANSLATION_PROVIDER: The translation service to use
    - TRANSLATION_TARGET_LANGUAGE: Default target language
    - TRANSLATION_CONFIDENCE_THRESHOLD: Minimum confidence for language detection
    - GOOGLE_TRANSLATE_API_KEY: API key for Google Translate
    - AZURE_TRANSLATOR_KEY: API key for Azure Translator
    - AZURE_TRANSLATOR_REGION: Region for Azure Translator
    """

    # Translation provider settings
    translation_provider: TranslationProviderType = "openai"
    target_language: str = "en"
    confidence_threshold: float = 0.8

    # Google Translate settings
    google_translate_api_key: Optional[str] = None
    google_project_id: Optional[str] = None

    # Azure Translator settings
    azure_translator_key: Optional[str] = None
    azure_translator_region: Optional[str] = None
    azure_translator_endpoint: str = "https://api.cognitive.microsofttranslator.com"

    # OpenAI uses the existing LLM configuration

    # Performance settings
    batch_size: int = 10
    max_retries: int = 3
    timeout_seconds: int = 30

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
        }


@lru_cache
def get_translation_config() -> TranslationConfig:
    """Get the translation configuration singleton."""
    return TranslationConfig()
