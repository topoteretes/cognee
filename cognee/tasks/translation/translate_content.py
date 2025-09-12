# pylint: disable=R0903, W0221
"""This module provides content translation capabilities for the Cognee framework."""
import asyncio
import os
from typing import Dict, Type, Protocol, Tuple, Optional

from cognee.shared.logging_utils import get_logger
from .models import TranslatedContent, LanguageMetadata

logger = get_logger()

# Custom exceptions for better error handling
class TranslationDependencyError(ImportError):
    """Raised when a required translation dependency is missing."""

class LangDetectError(TranslationDependencyError):
    """LangDetect library required."""

class OpenAIError(TranslationDependencyError):
    """OpenAI library required."""

class GoogleTranslateError(TranslationDependencyError):
    """GoogleTrans library required."""

class AzureTranslateError(TranslationDependencyError):
    """Azure AI Translation library required."""

class AzureConfigError(ValueError):
    """Azure configuration error."""

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

def _get_provider(translation_provider: str) -> TranslationProvider:
    """Returns a translation provider instance."""
    provider_class = _provider_registry.get(translation_provider.lower())
    if not provider_class:
        available = ', '.join(get_available_providers())
        msg = f"Unknown translation provider: {translation_provider}. Available providers: {available}"
        raise ValueError(msg)
    return provider_class()

# Built-in Providers
class NoOpProvider:
    """A provider that does nothing, used for testing or disabling translation."""
    async def detect_language(self, _text: str) -> Optional[Tuple[str, float]]:
        return None, 0.0

    async def translate(self, text: str, _target_language: str) -> Optional[Tuple[str, float]]:
        return text, 0.0

class LangDetectProvider:
    """
    A provider that uses the 'langdetect' library for offline language detection.
    This provider only detects the language and does not perform translation.
    """
    def __init__(self):
        try:
            from langdetect import detect_langs  # type: ignore[import-untyped]
            self._detect_langs = detect_langs
        except ImportError as e:
            raise LangDetectError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            detections = self._detect_langs(text)
        except Exception:
            logger.exception("Error during language detection")
            return None
        else:
            if not detections:
                return None
            best_detection = detections[0]
            return best_detection.lang, best_detection.prob

    async def translate(self, text: str, _target_language: str) -> Optional[Tuple[str, float]]:
        # This provider only detects language, does not translate.
        return text, 0.0

class OpenAIProvider:
    """A provider that uses OpenAI's API for translation."""
    def __init__(self):
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
            self.timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        except ImportError as e:
            raise OpenAIError() from e

    async def detect_language(self, _text: str) -> Optional[Tuple[str, float]]:
        # OpenAI's API does not have a separate language detection endpoint.
        # This can be implemented as part of the translation prompt if needed.
        return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_language}."},
                    {"role": "user", "content": text},
                ],
                max_tokens=1024,
                temperature=0.3,
                timeout=self.timeout,
            )
        except Exception:
            logger.exception("Error during OpenAI translation (model=%s)", self.model)
            return None
        else:
            translated_text = response.choices[0].message.content.strip()
            return translated_text, 1.0  # OpenAI does not provide a confidence score.

class GoogleTranslateProvider:
    """A provider that uses the 'googletrans' library for translation."""
    def __init__(self):
        try:
            from googletrans import Translator  # type: ignore[import-untyped]
            self.translator = Translator()
        except ImportError as e:
            raise GoogleTranslateError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            detection = await asyncio.to_thread(self.translator.detect, text)
        except Exception:
            logger.exception("Error during Google Translate language detection")
            return None
        else:
            return detection.lang, detection.confidence

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            translation = await asyncio.to_thread(self.translator.translate, text, dest=target_language)
        except Exception:
            logger.exception("Error during Google Translate translation")
            return None
        else:
            return translation.text, 1.0  # Confidence score not provided for translation.

class AzureTranslatorProvider:
    """A provider that uses Azure's Translator service."""
    def __init__(self):
        try:
            from azure.core.credentials import AzureKeyCredential  # type: ignore[import-untyped]
            from azure.ai.translation.text import TextTranslationClient  # type: ignore[import-untyped]
            
            self.key = os.getenv("AZURE_TRANSLATOR_KEY")
            self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com/")
            self.region = os.getenv("AZURE_TRANSLATOR_REGION", "global")

            if not self.key:
                raise AzureConfigError()

            self.client = TextTranslationClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
            )
        except ImportError as e:
            raise AzureTranslateError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            response = await asyncio.to_thread(self.client.detect, content=[text], country_hint=self.region)
        except Exception:
            logger.exception("Error during Azure language detection")
            return None
        else:
            detection = response[0].primary_language
            return detection.language, detection.score

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            response = await asyncio.to_thread(self.client.translate, content=[text], to=[target_language])
        except Exception:
            logger.exception("Error during Azure translation")
            return None
        else:
            translation = response[0].translations[0]
            return translation.text, 1.0  # Confidence score not provided for translation.

# Register built-in providers
register_translation_provider("noop", NoOpProvider)
register_translation_provider("langdetect", LangDetectProvider)
register_translation_provider("openai", OpenAIProvider)
register_translation_provider("google", GoogleTranslateProvider)
register_translation_provider("azure", AzureTranslatorProvider)

async def translate_content(  # pylint: disable=too-many-locals,too-many-branches
    *data_chunks,
    target_language: str = TARGET_LANGUAGE,
    translation_provider: str = "noop",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
):
    """
    Translate non-English chunks to the target language; attach language/translation metadata.
    Returns the (possibly modified) list of chunks.
    """
    provider = _get_provider(translation_provider)
    results = []
    
    # Support both pipeline varargs and a single list argument
    if len(data_chunks) == 1 and isinstance(data_chunks[0], list):
        _chunks = data_chunks[0]
    else:
        _chunks = list(data_chunks)
    
    for chunk in _chunks:
        text = getattr(chunk, "text", "") or ""
        content_id = getattr(chunk, "id", getattr(chunk, "chunk_index", "unknown"))
        
        # 1) Detect language
        try:
            detection = await provider.detect_language(text)
        except Exception:
            logger.exception("Language detection failed for content_id=%s", content_id)
            detection = None
            
        # Fallback: try 'langdetect' when the chosen provider can't detect
        if detection is None:
            fallback_cls = _provider_registry.get("langdetect")
            if fallback_cls is not None and not isinstance(provider, fallback_cls):
                try:
                    detection = await fallback_cls().detect_language(text)
                except Exception:
                    logger.exception("Fallback language detection failed for content_id=%s", content_id)
                    detection = None
                    
        # Normalize detection tuple; guard against (None, ""), bad types
        if detection is None:
            detected_language, conf = "unknown", 0.0
        else:
            lang_code, conf = detection
            if not isinstance(lang_code, str) or not lang_code.strip():
                detected_language, conf = "unknown", 0.0
            else:
                detected_language = lang_code.strip()

        # If detection is unavailable, allow translators to attempt translation.
        can_translate = translation_provider.lower() not in ("noop", "langdetect")
        requires = ((detected_language != target_language) and (conf >= confidence_threshold)) or (
            detected_language == "unknown" and can_translate and bool(text.strip())
        )
        
        # 2) Record language metadata
        chunk.metadata = getattr(chunk, "metadata", {}) or {}
        lang_meta = LanguageMetadata(
            content_id=str(content_id),
            detected_language=detected_language,
            language_confidence=conf,
            requires_translation=requires,
            character_count=len(text),
        )
        chunk.metadata["language"] = lang_meta.model_dump()
        
        # 3) Translate if required
        if requires:
            try:
                tr = await provider.translate(text, target_language)
            except Exception:
                logger.exception("Translation failed for content_id=%s", content_id)
                tr = None
                
            if tr:
                translated_text, t_conf = tr
                if translated_text != text:
                    trans = TranslatedContent(
                        original_chunk_id=str(content_id),
                        original_text=text,
                        translated_text=translated_text,
                        source_language=detected_language,
                        target_language=target_language,
                        translation_provider=translation_provider.lower(),
                        confidence_score=t_conf or 0.0,
                    )
                    chunk.metadata["translation"] = trans.model_dump()
                    chunk.text = translated_text
                else:
                    logger.info("Provider returned unchanged text; skipping translation metadata (content_id=%s)", content_id)
            else:
                # Translation call failed (exception or None) — record a no-op entry
                trans = TranslatedContent(
                    original_chunk_id=str(content_id),
                    original_text=text,
                    translated_text=text,
                    source_language=detected_language,
                    target_language=target_language,
                    translation_provider=translation_provider.lower(),
                    confidence_score=0.0,
                )
                chunk.metadata["translation"] = trans.model_dump()
                
        results.append(chunk)
    return results
