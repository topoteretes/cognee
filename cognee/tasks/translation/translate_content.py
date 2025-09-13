# pylint: disable=R0903, W0221
"""This module provides content translation capabilities for the Cognee framework."""
import asyncio
import math
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
try:
    CONFIDENCE_THRESHOLD = float(os.getenv("COGNEE_TRANSLATION_CONFIDENCE_THRESHOLD", "0.80"))
except (TypeError, ValueError):
    logger.warning(
        "Invalid float for COGNEE_TRANSLATION_CONFIDENCE_THRESHOLD=%r; defaulting to 0.80",
        os.getenv("COGNEE_TRANSLATION_CONFIDENCE_THRESHOLD"),
    )
    CONFIDENCE_THRESHOLD = 0.80

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
# Helpers
def _normalize_lang_code(code: Optional[str]) -> str:
    if not isinstance(code, str) or not code.strip():
        return "unknown"
    c = code.strip().replace("_", "-")
    parts = c.split("-")
    if len(parts) == 1 and len(parts[0]) == 2 and parts[0].isalpha():
        return parts[0].lower()
    if len(parts) >= 2 and len(parts[0]) == 2 and parts[1]:
        return f"{parts[0].lower()}-{parts[1][:2].upper()}"
    return "unknown"

async def _detect_language_with_fallback(provider: TranslationProvider, text: str, content_id: str) -> tuple[str, float]:
    try:
        detection = await provider.detect_language(text)
    except Exception:
        logger.exception("Language detection failed for content_id=%s", content_id)
        detection = None

    if detection is None:
        fallback_cls = _provider_registry.get("langdetect")
        if fallback_cls is not None and not isinstance(provider, fallback_cls):
            try:
                detection = await fallback_cls().detect_language(text)
            except Exception:
                logger.exception("Fallback language detection failed for content_id=%s", content_id)
                detection = None

    if detection is None:
        return "unknown", 0.0

    lang_code, conf = detection
    detected_language = _normalize_lang_code(lang_code)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    if math.isnan(conf):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return detected_language, conf

def _requires_translation(detected_language: str, target_language: str, conf: float, threshold: float, provider_name: str, text: str) -> bool:
    can_translate = provider_name not in ("noop", "langdetect")
    if detected_language == "unknown":
        return can_translate and bool(text.strip())
    return (detected_language != target_language) and (conf >= threshold)

def _attach_language_metadata(chunk, content_id: str, detected_language: str, conf: float, requires: bool, text: str) -> None:
    chunk.metadata = getattr(chunk, "metadata", {}) or {}
    lang_meta = LanguageMetadata(
        content_id=str(content_id),
        detected_language=detected_language,
        language_confidence=conf,
        requires_translation=requires,
        character_count=len(text),
    )
    chunk.metadata["language"] = lang_meta.model_dump()

async def _translate_and_update(provider: TranslationProvider, chunk, content_id: str, text: str, target_language: str, detected_language: str) -> None:
    try:
        tr = await provider.translate(text, target_language)
    except Exception:
        logger.exception("Translation failed for content_id=%s", content_id)
        tr = None

    if tr and isinstance(tr[0], str) and tr[0].strip() and tr[0] != text:
        translated_text, t_conf = tr
        provider_used = provider.__class__.__name__.replace("Provider", "").lower()
        trans = TranslatedContent(
            original_chunk_id=str(content_id),
            original_text=text,
            translated_text=translated_text,
            source_language=detected_language,
            target_language=target_language,
            translation_provider=provider_used,
            confidence_score=t_conf or 0.0,
        )
        chunk.metadata["translation"] = trans.model_dump()
        chunk.text = translated_text
        if hasattr(chunk, "chunk_size"):
            try:
                chunk.chunk_size = len(translated_text.split())
            except (AttributeError, ValueError, TypeError):
                logger.debug(
                    "Could not update chunk_size for content_id=%s",
                    content_id,
                    exc_info=True,
                )
        return

    if tr is None:
        provider_used = provider.__class__.__name__.replace("Provider", "").lower()
        trans = TranslatedContent(
            original_chunk_id=str(content_id),
            original_text=text,
            translated_text=text,
            source_language=detected_language,
            target_language=target_language,
            translation_provider=provider_used,
            confidence_score=0.0,
        )
        chunk.metadata["translation"] = trans.model_dump()
    else:
        logger.info("Provider returned unchanged text; skipping translation metadata (content_id=%s)", content_id)


# Test helpers for registry isolation
def snapshot_registry() -> Dict[str, Type[TranslationProvider]]:
    """Return a shallow copy snapshot of the provider registry (for tests)."""
    return dict(_provider_registry)

def restore_registry(snapshot: Dict[str, Type[TranslationProvider]]) -> None:
    """Restore the provider registry from a snapshot (for tests)."""
    _provider_registry.clear()
    _provider_registry.update(snapshot)

def validate_provider(name: str) -> None:
    """Ensure a provider can be resolved and instantiated or raise."""
    _get_provider(name)

# Built-in Providers
class NoOpProvider:
    """A provider that does nothing, used for testing or disabling translation."""
    async def detect_language(self, _text: str) -> Optional[Tuple[str, float]]:
        return None

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
            response = await self.client.with_options(timeout=self.timeout).chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_language}."},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
            )
        except Exception:
            logger.exception("Error during OpenAI translation (model=%s)", self.model)
            return None
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text, 0.0  # No calibrated confidence available.

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

        try:
            conf = float(detection.confidence) if detection.confidence is not None else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        return detection.lang, conf

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            translation = await asyncio.to_thread(self.translator.translate, text, dest=target_language)
        except Exception:
            logger.exception("Error during Google Translate translation")
            return None
        
        return translation.text, 0.0  # Confidence not provided.

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
        
        detection = response[0].primary_language
        return detection.language, detection.score

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            response = await asyncio.to_thread(self.client.translate, content=[text], to=[target_language])
        except Exception:
            logger.exception("Error during Azure translation")
            return None
        
        translation = response[0].translations[0]
        return translation.text, 0.0  # Confidence not provided.

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

    Batching behavior:
    - Accepts either varargs of chunk objects (pipeline may pass multiple args),
      or a single list containing chunk objects. Both forms are supported.
    """
    provider = _get_provider(translation_provider)
    results = []

    if len(data_chunks) == 1 and isinstance(data_chunks[0], list):
        _chunks = data_chunks[0]
    else:
        _chunks = list(data_chunks)

    for chunk in _chunks:
        text = getattr(chunk, "text", "") or ""
        content_id = getattr(chunk, "id", getattr(chunk, "chunk_index", "unknown"))
        detected_language, conf = await _detect_language_with_fallback(provider, text, str(content_id))
        requires = _requires_translation(
            detected_language,
            target_language,
            conf,
            confidence_threshold,
            translation_provider.lower(),
            text,
        )
        _attach_language_metadata(chunk, str(content_id), detected_language, conf, requires, text)
        if requires:
            await _translate_and_update(provider, chunk, str(content_id), text, target_language, detected_language)
        results.append(chunk)
    return results
    return results
