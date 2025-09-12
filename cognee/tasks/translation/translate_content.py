# pylint: disable=R0903, W0221
"""This module provides content translation capabilities for the Cognee framework."""
import asyncio
import os
from typing import Dict, Type, Protocol, Tuple, Optional

from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from .models import TranslatedContent, LanguageMetadata

logger = get_logger()

# Environment variables for configuration
TARGET_LANGUAGE = os.getenv("COGNEE_TRANSLATION_TARGET_LANGUAGE", "en")
CONFIDENCE_THRESHOLD = float(os.getenv("COGNEE_TRANSLATION_CONFIDENCE_THRESHOLD", 0.80))

class TranslationProvider(Protocol):
    """Protocol for translation providers."""
    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        """Detects the language of the given text."""

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        """Translates the given text to the target language."""

# Registry for translation providers
_provider_registry: Dict[str, Type[TranslationProvider]] = {}

def register_translation_provider(name: str, provider: Type[TranslationProvider]):
    """Registers a new translation provider."""
    _provider_registry[name.lower()] = provider

def get_available_providers():
    """Returns a list of available translation providers."""
    return list(_provider_registry.keys())

def _get_provider(provider_name: str) -> TranslationProvider:
    """Returns a translation provider instance."""
    provider_class = _provider_registry.get(provider_name.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown translation provider: {provider_name}. "
            f"Available providers: {', '.join(get_available_providers())}"
        )
    return provider_class()

# Built-in Providers
class NoOpProvider:
    """A provider that does nothing, used for testing or disabling translation."""
    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        return None, 0.0

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        return text, 0.0

class LangDetectProvider:
    """
    A provider that uses the 'langdetect' library for offline language detection.
    This provider only detects the language and does not perform translation.
    """
    def __init__(self):
        try:
            from langdetect import detect_langs
            self._detect_langs = detect_langs
        except ImportError as e:
            raise ImportError(
                "The 'langdetect' library is required for LangDetectProvider. "
                "Please install it using: pip install langdetect"
            ) from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            detections = self._detect_langs(text)
            if not detections:
                return None
            best_detection = detections[0]
            return best_detection.lang, best_detection.prob
        except Exception as e:
            logger.error("Error during language detection: %s", e)
            return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        # This provider only detects language, does not translate.
        return text, 0.0

class OpenAIProvider:
    """A provider that uses OpenAI's API for translation."""
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError as e:
            raise ImportError(
                "The 'openai' library is required for OpenAIProvider. "
                "Please install it using: pip install openai"
            ) from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        # OpenAI's API does not have a separate language detection endpoint.
        # This can be implemented as part of the translation prompt if needed.
        return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_language}."},
                    {"role": "user", "content": text},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            translated_text = response.choices[0].message.content.strip()
            return translated_text, 1.0  # OpenAI does not provide a confidence score.
        except Exception as e:
            logger.error("Error during OpenAI translation: %s", e)
            return None

class GoogleTranslateProvider:
    """A provider that uses the 'googletrans' library for translation."""
    def __init__(self):
        try:
            from googletrans import Translator
            self.translator = Translator()
        except ImportError as e:
            raise ImportError(
                "The 'googletrans' library is required for GoogleTranslateProvider. "
                "Please install it using: pip install googletrans==4.0.0rc1"
            ) from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            detection = self.translator.detect(text)
            return detection.lang, detection.confidence
        except Exception as e:
            logger.error("Error during Google Translate language detection: %s", e)
            return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            translation = self.translator.translate(text, dest=target_language)
            return translation.text, 1.0  # Confidence score not provided for translation.
        except Exception as e:
            logger.error("Error during Google Translate translation: %s", e)
            return None

class AzureTranslatorProvider:
    """A provider that uses Azure's Translator service."""
    def __init__(self):
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.ai.translation.text import TextTranslationClient
            
            self.key = os.getenv("AZURE_TRANSLATOR_KEY")
            self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com/")
            self.region = os.getenv("AZURE_TRANSLATOR_REGION", "global")

            if not self.key:
                raise ValueError("AZURE_TRANSLATOR_KEY is not set.")

            self.client = TextTranslationClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
            )
        except ImportError as e:
            raise ImportError(
                "The 'azure-ai-translation-text' library is required for AzureTranslatorProvider. "
                "Please install it using: pip install azure-ai-translation-text"
            ) from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            response = self.client.detect(content=[text], country_hint=self.region)
            detection = response[0].primary_language
            return detection.language, detection.score
        except Exception as e:
            logger.error("Error during Azure language detection: %s", e)
            return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            response = self.client.translate(content=[text], to=[target_language])
            translation = response[0].translations[0]
            return translation.text, 1.0  # Confidence score not provided for translation.
        except Exception as e:
            logger.error("Error during Azure translation: %s", e)
            return None

# Register built-in providers
register_translation_provider("noop", NoOpProvider)
register_translation_provider("langdetect", LangDetectProvider)
register_translation_provider("openai", OpenAIProvider)
register_translation_provider("google", GoogleTranslateProvider)
register_translation_provider("azure", AzureTranslatorProvider)

async def translate_content(
    graph: KnowledgeGraph,
    provider_name: str = "noop",
    target_language: str = TARGET_LANGUAGE,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> KnowledgeGraph:
    """
    Translate the content of a KnowledgeGraph if it's not in the target language.
    """
    provider = _get_provider(provider_name)
    original_content = graph.source_content

    # 1. Detect language
    detection_result = await provider.detect_language(original_content)
    if detection_result is None:
        logger.warning("Language detection failed. Skipping translation.")
        return graph

    detected_language, confidence = detection_result
    
    # 2. Check if translation is needed
    if detected_language == target_language or confidence < confidence_threshold:
        logger.info(
            "Skipping translation for content (lang=%s, conf=%.2f)",
            detected_language,
            confidence,
        )
        return graph

    logger.info(
        "Translating content from '%s' to '%s' (confidence: %.2f)",
        detected_language,
        target_language,
        confidence,
    )

    # 3. Translate content
    translation_result = await provider.translate(original_content, target_language)
    if translation_result is None:
        logger.error("Translation failed. Using original content.")
        return graph

    translated_text, translation_confidence = translation_result

    # 4. Store translation details in metadata
    graph.translated_content = TranslatedContent(
        translated_text=translated_text,
        original_language=LanguageMetadata(
            language=detected_language,
            confidence=confidence,
        ),
        translation_confidence=translation_confidence,
        provider=provider_name,
    )

    # 5. Replace source_content with translated_text for subsequent pipeline steps
    graph.source_content = translated_text
    
    logger.info("Content translated successfully. Provider: %s", provider_name)
    return graph
