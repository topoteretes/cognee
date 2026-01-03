"""
Base classes for translation providers.

This module defines the abstract interface that all translation providers must implement.
Providers handle the actual translation of text using external services like OpenAI,
Google Translate, or Azure Translator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranslationResult:
    """Result of a translation operation."""

    translated_text: str
    source_language: str
    target_language: str
    # Confidence score from the provider, or None if not available (e.g., Google Translate)
    confidence_score: Optional[float]
    provider: str
    raw_response: Optional[dict] = None


class TranslationProvider(ABC):
    """Abstract base class for translation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this translation provider."""
        pass

    @abstractmethod
    async def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text to the target language.

        Args:
            text: The text to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (optional, will be auto-detected if not provided)

        Returns:
            TranslationResult with translated text and metadata
        """
        pass

    @abstractmethod
    async def translate_batch(
        self,
        texts: list[str],
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts to the target language.

        Args:
            texts: List of texts to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (optional)

        Returns:
            List of TranslationResult objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (has required credentials).

        All providers must implement this method to validate their credentials.

        Returns:
            True if the provider has valid credentials and is ready to use.
        """
        pass
