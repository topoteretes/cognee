# pylint: disable=R0903, W0221
"""This module provides content translation capabilities for the Cognee framework."""
import asyncio
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional
from cognee.shared.logging_utils import get_logger
from .models import TranslatedContent, LanguageMetadata
from .translation_providers_enum import TranslationProviderEnum, TranslationProvider
from .translation_registry import (
    register_translation_provider,
    get_available_providers,
    get_provider_class,
    snapshot_registry,
    restore_registry,
    validate_provider,
)
from .translation_providers.llm_provider import LLMProvider
from .translation_providers.google_provider import GoogleTranslateProvider
from .translation_providers.azure_provider import AzureTranslateProvider
from .translation_providers.langdetect_provider import LangDetectProvider
from .translation_providers.noop_provider import NoopProvider

logger = get_logger(__name__)


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


def _normalize_confidence(confidence: Any) -> float:
    """Normalize confidence value to float in [0.0, 1.0] range."""
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


@dataclass
class TranslationContext:  # pylint: disable=too-many-instance-attributes
    """A context object to hold data for a single translation operation."""
    provider: TranslationProvider
    chunk: Any
    text: str
    target_language: str
    confidence_threshold: float
    content_id: str = field(init=False)
    detected_language: str = "unknown"
    detection_confidence: float = 0.0
    requires_translation: bool = False

    def __post_init__(self):
        # Try to set content_id from chunk attributes
        for attr in ("id", "content_id", "uuid", "pk"):
            if hasattr(self.chunk, attr):
                self.content_id = str(getattr(self.chunk, attr))
                break
        else:
            self.content_id = "unknown"

from .translation_providers_enum import TranslationProviderEnum, TranslationProvider
from .translation_registry import (
    register_translation_provider,
    get_available_providers,
    get_provider_class,
    snapshot_registry,
    restore_registry,
    validate_provider,
)
from .translation_providers.llm_provider import LLMProvider
from .translation_providers.google_provider import GoogleTranslateProvider
from .translation_providers.azure_provider import AzureTranslateProvider
from .translation_providers.langdetect_provider import LangDetectProvider
from .translation_providers.noop_provider import NoopProvider

class TranslationProviderError(ValueError):
    """Error related to translation provider initialization."""
    pass

class UnknownTranslationProviderError(TranslationProviderError):
    """Unknown translation provider name."""

class ProviderInitializationError(TranslationProviderError):
    """Provider failed to initialize (likely missing dependency or bad config)."""

def _get_provider(translation_provider: str) -> TranslationProvider:
    """Resolve and instantiate a registered translation provider by name."""
    provider_cls = get_provider_class(translation_provider)
    return provider_cls()

def _normalize_lang_code(code: Optional[str]) -> str:
    """Normalize a language code to a canonical form or return 'unknown'."""
    if not isinstance(code, str) or not code.strip():
        return "unknown"
    c = code.strip().replace("_", "-")
    parts = c.split("-")
    lang = parts[0]
    if len(lang) == 2 and lang.isalpha():
        if len(parts) >= 2:
            region = parts[1]
            if len(region) == 2 and region.isalpha():
                return f"{lang.lower()}-{region.upper()}"
    return lang.lower()

async def _detect_language_with_fallback(provider: TranslationProvider, text: str, content_id: str) -> Tuple[str, float]:
    try:
        detection = await provider.detect_language(text)
    except Exception:
        detection = None
    if detection is None:
        try:
            fallback_cls = get_provider_class("langdetect")
            fallback_provider = fallback_cls()
            if not isinstance(provider, fallback_cls):
                detection = await fallback_provider.detect_language(text)
        except Exception:
            detection = None
    if detection is None:
        return "unknown", 0.0
    lang_code, confidence = detection
    detected_language = _normalize_lang_code(lang_code)
    confidence = _normalize_confidence(confidence)
    return detected_language, confidence

def _decide_if_translation_is_required(translation_context: TranslationContext) -> None:
    """Decide whether a translation should be performed and update translation_context.requires_translation."""
    target_language = _normalize_lang_code(translation_context.target_language)

    if translation_context.detected_language == "unknown":
        # Decide purely from input; provider capability is handled via fallbacks.
        translation_context.requires_translation = bool(translation_context.text.strip())
    else:
        same_base = translation_context.detected_language.split("-")[0] == target_language.split("-")[0]
        translation_context.requires_translation = (
            (not same_base) and translation_context.detection_confidence >= translation_context.confidence_threshold
        )

def _attach_language_metadata(translation_context: TranslationContext) -> None:
    """Attach language detection and translation decision metadata to the context's chunk."""
    translation_context.chunk.metadata = getattr(translation_context.chunk, "metadata", {}) or {}
    lang_meta = LanguageMetadata(
        content_id=str(translation_context.content_id),
        detected_language=translation_context.detected_language,
        language_confidence=translation_context.detection_confidence,
        requires_translation=translation_context.requires_translation,
        character_count=len(translation_context.text),
    )
    translation_context.chunk.metadata["language"] = lang_meta.model_dump()

def _build_provider_plan(translation_provider_name, fallback_input):
    primary_key = (translation_provider_name or "noop").lower()
    raw = fallback_input or []
    fallback_providers = []
    seen = {primary_key}
    invalid_providers = []
    available_providers = set(get_available_providers())
    for p in raw:
        if isinstance(p, str) and p.strip():
            key = p.strip().lower()
            if key in available_providers and key not in seen:
                fallback_providers.append(key)
                seen.add(key)
            else:
                invalid_providers.append(p)
    if invalid_providers:
        logger.warning("Ignoring unknown fallback providers: %s", invalid_providers)
    return primary_key, fallback_providers


async def _translate_and_update(translation_context: TranslationContext) -> None:
    """Translate the text in translation_context and update the chunk's metadata if successful."""
    try:
        result = await translation_context.provider.translate(
            translation_context.text, translation_context.target_language
        )
    except Exception as e:
        logger.exception("Translation failed: %s", e)
        return
    if result is not None:
        translated_text, confidence = result
        if translated_text and translated_text != translation_context.text:
            translation_context.chunk.text = translated_text
            translation_context.chunk.metadata = getattr(translation_context.chunk, "metadata", {}) or {}
            translation_context.chunk.metadata["translation"] = {
                "provider": type(translation_context.provider).__name__,
                "confidence": confidence,
                "translated_text": translated_text,
            }

async def _process_chunk(chunk, plan, provider_cache):
    # Unpack plan: (target_language, primary_key, fallback_providers, confidence_threshold, detection_provider_name)
    target_language, primary_key, fallback_providers, confidence_threshold, detection_provider_name = plan
    try:
        provider = provider_cache.get(primary_key)
        if provider is None:
            provider = _get_provider(primary_key)
            provider = provider_cache.setdefault(primary_key, provider)
    except asyncio.CancelledError:
        raise
    except (ImportError, ValueError) as e:
        logger.error("Provider import/value error for %s: %s", primary_key, e)
        return chunk
    except Exception as e:
        logger.exception("Failed to initialize translation provider: %s", primary_key)
        return chunk

    text_to_translate = getattr(chunk, "text", "")
    if not isinstance(text_to_translate, str) or not text_to_translate.strip():
        return chunk

    translation_context = TranslationContext(
        provider=provider,
        chunk=chunk,
        text=text_to_translate,
        target_language=target_language,
        confidence_threshold=confidence_threshold,
    )

    # Attempt detection using the requested detection provider; fall back to the provider's detection or langdetect
    detection = None
    try:
        detector_cls = get_provider_class(detection_provider_name)
        detector = detector_cls()
        if hasattr(detector, "detect_language"):
            detection = await detector.detect_language(text_to_translate)
    except Exception:
        detection = None

    if detection is None:
        # Fallback to original detection-with-fallback semantics
        translation_context.detected_language, translation_context.detection_confidence = await _detect_language_with_fallback(
            provider, text_to_translate, str(translation_context.content_id)
        )
    else:
        lang_code, confidence = detection
        translation_context.detected_language = _normalize_lang_code(lang_code)
        translation_context.detection_confidence = _normalize_confidence(confidence)

    _decide_if_translation_is_required(translation_context)
    _attach_language_metadata(translation_context)

    if translation_context.requires_translation:
        # Short-circuit: primary provider cannot translate and no fallbacks provided
        if primary_key == "noop" and not fallback_providers:
            return translation_context.chunk
        await _translate_and_update(translation_context)
        # If no translation metadata was produced, try fallbacks in order
        if "translation" not in getattr(translation_context.chunk, "metadata", {}):
            for alternative_provider_name in fallback_providers:
                try:
                    alternative_provider = provider_cache.get(alternative_provider_name)
                    if alternative_provider is None:
                        alternative_provider = _get_provider(alternative_provider_name)
                        alternative_provider = provider_cache.setdefault(alternative_provider_name, alternative_provider)
                except asyncio.CancelledError:
                    raise
                except (ImportError, ValueError) as e:
                    logger.error("Fallback provider import/value error for %s: %s", alternative_provider_name, e)
                    continue
                except Exception as e:
                    logger.exception("Failed to initialize fallback translation provider: %s", alternative_provider_name)
                    continue
                translation_context.provider = alternative_provider
                await _translate_and_update(translation_context)
                if "translation" in getattr(translation_context.chunk, "metadata", {}):
                    break

    return translation_context.chunk


async def translate_content(*chunks: Any, **kwargs) -> Any:
    """
    Translate the content of a chunk if necessary.

    This function detects the language of the chunk's text, decides if translation is needed,
    and if so, translates the text to the target language using the specified provider.
    It updates the chunk with the translated text and adds metadata about the translation process.
    
    Args:
        *chunks: The chunk(s) of content to be processed. Each chunk must have a 'text' attribute.
                Can be called as:
                - translate_content(chunk) - single chunk
                - translate_content(chunk1, chunk2, ...) - multiple chunks
                - translate_content([chunk1, chunk2, ...]) - list of chunks
        **kwargs: Additional arguments:
            target_language (str): Target language code (default from COGNEE_TRANSLATION_TARGET_LANGUAGE).
            translation_provider (str): Primary provider key (e.g., "openai", "google", "azure", "noop").
                                      Defaults to "noop".
            fallback_providers (List[str]): Ordered list of provider keys to try if the primary 
                                          fails or returns unchanged text. Defaults to empty list.
            confidence_threshold (float): Minimum confidence threshold for language detection 
                                        (default from COGNEE_TRANSLATION_CONFIDENCE_THRESHOLD).
    
    Returns:
        Any: For single chunk input - returns the processed chunk directly.
             For multiple chunks input - returns List[Any] of processed chunks.
             Each returned chunk may have its text translated and metadata updated with:
             - language: detected language and confidence
             - translation: translated text and provider information (if translation occurred)
    """
    # Always work with a list internally for consistency
    if len(chunks) == 1 and isinstance(chunks[0], list):
        # Single list argument: translate_content([chunk1, chunk2, ...])
        batch = chunks[0]
        return_single = False
    elif len(chunks) == 1:
        # Single chunk argument: translate_content(chunk)
        batch = list(chunks)
        return_single = True
    else:
        # Multiple chunk arguments: translate_content(chunk1, chunk2, ...)
        batch = list(chunks)
        return_single = False

    target_language = kwargs.get("target_language", TARGET_LANGUAGE)
    translation_provider_name = kwargs.get("translation_provider", "noop")
    primary_key, fallback_providers = _build_provider_plan(
        translation_provider_name, kwargs.get("fallback_providers", [])
    )
    detection_provider_name = kwargs.get("detection_provider", "langdetect")
    confidence_threshold = kwargs.get("confidence_threshold", CONFIDENCE_THRESHOLD)

    # Provider cache for this batch to reduce instantiation overhead
    provider_cache: Dict[str, Any] = {}
    
    # Bundle plan parameters to reduce argument count
    plan = (target_language, primary_key, fallback_providers, confidence_threshold, detection_provider_name)
    
    # Parse concurrency with error handling
    try:
        max_concurrency = int(os.getenv("COGNEE_TRANSLATION_MAX_CONCURRENCY", "8"))
    except (TypeError, ValueError):
        logger.warning("Invalid COGNEE_TRANSLATION_MAX_CONCURRENCY; defaulting to 8")
        max_concurrency = 8
    if max_concurrency < 1:
        logger.warning("COGNEE_TRANSLATION_MAX_CONCURRENCY < 1; clamping to 1")
        max_concurrency = 1
    
    sem = asyncio.Semaphore(max_concurrency)
    async def _wrapped(c):
        async with sem:
            return await _process_chunk(c, plan, provider_cache)
    results = await asyncio.gather(*(_wrapped(c) for c in batch))

    return results[0] if return_single else results


# Initialize providers
register_translation_provider("noop", NoopProvider)
register_translation_provider("langdetect", LangDetectProvider)
register_translation_provider("llm", LLMProvider)
register_translation_provider("google", GoogleTranslateProvider)
register_translation_provider("azure", AzureTranslateProvider)