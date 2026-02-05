"""
Translation task for Cognee.

This module provides multilingual content translation capabilities,
allowing automatic detection and translation of non-English content
to a target language while preserving original text and metadata.

Main Components:
- translate_content: Main task function for translating document chunks
- translate_text: Convenience function for translating single texts
- batch_translate_texts: Batch translation for multiple texts
- detect_language: Language detection utility
- TranslatedContent: DataPoint model for translated content
- LanguageMetadata: DataPoint model for language information

Supported Translation Providers:
- LLM (default): Uses the configured LLM via existing infrastructure
- Google Translate: Requires google-cloud-translate package
- Azure Translator: Requires Azure Translator API key

Example Usage:
    ```python
    from cognee.tasks.translation import translate_content, translate_text

    # Translate document chunks in a pipeline
    translated_chunks = await translate_content(
        chunks,
        target_language="en",
        translation_provider="llm"
    )

    # Translate a single text
    result = await translate_text("Bonjour le monde!")
    print(result.translated_text)  # "Hello world!"
    ```
"""

from .config import get_translation_config, TranslationConfig
from .detect_language import (
    detect_language,
    detect_language_async,
    LanguageDetectionResult,
    get_language_name,
)
from .exceptions import (
    TranslationError,
    LanguageDetectionError,
    TranslationProviderError,
    UnsupportedLanguageError,
    TranslationConfigError,
)
from .models import TranslatedContent, LanguageMetadata
from .providers import (
    TranslationProvider,
    TranslationResult,
    get_translation_provider,
    LLMTranslationProvider,
    GoogleTranslationProvider,
    AzureTranslationProvider,
)
from .translate_content import (
    translate_content,
    translate_text,
    batch_translate_texts,
)

__all__ = [
    # Main task functions
    "translate_content",
    "translate_text",
    "batch_translate_texts",
    # Language detection
    "detect_language",
    "detect_language_async",
    "LanguageDetectionResult",
    "get_language_name",
    # Models
    "TranslatedContent",
    "LanguageMetadata",
    # Configuration
    "get_translation_config",
    "TranslationConfig",
    # Providers
    "TranslationProvider",
    "TranslationResult",
    "get_translation_provider",
    "LLMTranslationProvider",
    "GoogleTranslationProvider",
    "AzureTranslationProvider",
    # Exceptions
    "TranslationError",
    "LanguageDetectionError",
    "TranslationProviderError",
    "UnsupportedLanguageError",
    "TranslationConfigError",
]
