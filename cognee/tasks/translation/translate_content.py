# pylint: disable=R0903, W0221
"""This module provides content translation capabilities for the Cognee framework."""
import asyncio
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Type, Protocol, Tuple, Optional

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


@dataclass
class TranslationContext:
    """A context object to hold data for a single translation operation."""
    provider: "TranslationProvider"
    chunk: Any
    text: str
    target_language: str
    confidence_threshold: float
    provider_name: str
    content_id: str = field(init=False)
    detected_language: str = "unknown"
    detection_confidence: float = 0.0
    requires_translation: bool = False

    def __post_init__(self):
        """
        Initialize derived fields after dataclass construction.
        
        Sets self.content_id to the first available identifier on self.chunk in this order:
        - self.chunk.id
        - self.chunk.chunk_index
        If neither attribute exists, content_id is set to the string "unknown".
        """
        self.content_id = getattr(self.chunk, "id", getattr(self.chunk, "chunk_index", "unknown"))


class TranslationProvider(Protocol):
    """Protocol for translation providers."""
    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect the language of the provided text.
        
        Uses the langdetect library to determine the most likely language and its probability.
        Returns a tuple (language_code, confidence) where `language_code` is a normalized short code (e.g., "en", "fr" or "unknown") and `confidence` is a float in [0.0, 1.0]. Returns None when detection fails (empty input, an error, or no reliable result).
        """

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        """
        Translate the given text into the specified target language asynchronously.
        
        Parameters:
            text: The source text to translate.
            target_language: Target language code (e.g., "en", "es", "fr-CA").
        
        Returns:
            A tuple (translated_text, confidence) on success, where `confidence` is a float in [0.0, 1.0] (may be 0.0 if the provider does not supply a score), or None if translation failed or was unavailable.
        """

# Registry for translation providers
_provider_registry: Dict[str, Type[TranslationProvider]] = {}

def register_translation_provider(name: str, provider: Type[TranslationProvider]):
    """
    Register a translation provider under a canonical lowercase key.
    
    The provided class will be stored in the internal provider registry and looked up by its lowercased `name`. If an entry with the same key already exists it will be replaced.
    
    Parameters:
        name (str): Human-readable provider name (case-insensitive); stored as lower-case.
        provider (Type[TranslationProvider]): Provider class implementing the TranslationProvider protocol; instances are constructed when the provider is resolved.
    """
    _provider_registry[name.lower()] = provider

def get_available_providers():
    """Returns a list of available translation providers."""
    return sorted(_provider_registry.keys())

def _get_provider(translation_provider: str) -> TranslationProvider:
    """
    Resolve and instantiate a registered translation provider by name.
    
    The lookup is case-insensitive: `translation_provider` should be the provider key (e.g., "openai", "google", "noop").
    Returns an instance of the provider implementing the TranslationProvider protocol.
    
    Raises:
        ValueError: if no provider is registered under the given name; the error message lists available providers.
    """
    provider_class = _provider_registry.get(translation_provider.lower())
    if not provider_class:
        available = ', '.join(get_available_providers())
        msg = f"Unknown translation provider: {translation_provider}. Available providers: {available}"
        raise ValueError(msg)
    return provider_class()
# Helpers
def _normalize_lang_code(code: Optional[str]) -> str:
    """
    Normalize a language code to a canonical form or return "unknown".
    
    Normalizes common language code formats:
    - Two-letter codes (e.g., "en", "EN", " en ") -> "en"
    - Locale codes with region (e.g., "en-us", "en_US", "EN-us") -> "en-US"
    - Returns "unknown" for empty, non-string, or unrecognized inputs.
    
    Parameters:
        code (Optional[str]): Language code or locale string to normalize.
    
    Returns:
        str: Normalized language code in either "xx" or "xx-YY" form, or "unknown" if input is invalid.
    """
    if not isinstance(code, str) or not code.strip():
        return "unknown"
    c = code.strip().replace("_", "-")
    parts = c.split("-")
    if len(parts) == 1 and len(parts[0]) == 2 and parts[0].isalpha():
        return parts[0].lower()
    if len(parts) >= 2 and len(parts[0]) == 2 and parts[1]:
        return f"{parts[0].lower()}-{parts[1][:2].upper()}"
    return "unknown"

def _provider_name(provider: TranslationProvider) -> str:
    """Return the canonical registry key for a provider instance, or a best-effort name."""
    return next(
        (name for name, cls in _provider_registry.items() if isinstance(provider, cls)),
        provider.__class__.__name__.replace("Provider", "").lower(),
    )

async def _detect_language_with_fallback(provider: TranslationProvider, text: str, content_id: str) -> Tuple[str, float]:
    """
    Detect the language of `text`, falling back to the registered "langdetect" provider if the primary provider fails.
    
    Attempts to call the primary provider's `detect_language`. If that call returns None or raises, and a different "langdetect" provider is registered, it will try the fallback. Detection failures are logged; exceptions are not propagated.
    
    Parameters:
        text (str): The text to detect language for.
        content_id (str): Identifier used in logs to correlate errors to the input content.
    
    Returns:
        Tuple[str, float]: A normalized language code (e.g., "en" or "pt-BR") and a confidence score in [0.0, 1.0].
        On detection failure returns ("unknown", 0.0). Confidence values are coerced to float, NaNs converted to 0.0, and clamped to the [0.0, 1.0] range.
    """
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

def _decide_if_translation_is_required(ctx: TranslationContext) -> None:
    """
    Decide whether a translation should be performed and update ctx.requires_translation.
    
    Normalizes the configured target language and marks translation as required only when:
    - The provider can perform translations (not "noop" or "langdetect"), and
    - Either the detected language is "unknown" and the text is non-empty, or
    - The detected language (normalized) differs from the target language and the detection confidence meets or exceeds ctx.confidence_threshold.
    
    The function mutates the provided TranslationContext in-place and does not return a value.
    """
    # Normalize to align with detected_language normalization and model regex.
    target_language = _normalize_lang_code(ctx.target_language)
    can_translate = ctx.provider_name not in ("noop", "langdetect")

    if ctx.detected_language == "unknown":
        ctx.requires_translation = can_translate and bool(ctx.text.strip())
    else:
        ctx.requires_translation = (
            ctx.detected_language != target_language
            and ctx.detection_confidence >= ctx.confidence_threshold
        )

def _attach_language_metadata(ctx: TranslationContext) -> None:
    """
    Attach language detection and translation decision metadata to the context's chunk.
    
    Ensures the chunk has a metadata mapping, builds a LanguageMetadata record from
    the context (content_id, detected language and confidence, whether translation is
    required, and character count of the text), serializes it, and stores it under
    the "language" key in chunk.metadata.
    
    Parameters:
        ctx (TranslationContext): Context containing the chunk and detection/decision values.
    """
    ctx.chunk.metadata = getattr(ctx.chunk, "metadata", {}) or {}
    lang_meta = LanguageMetadata(
        content_id=str(ctx.content_id),
        detected_language=ctx.detected_language,
        language_confidence=ctx.detection_confidence,
        requires_translation=ctx.requires_translation,
        character_count=len(ctx.text),
    )
    ctx.chunk.metadata["language"] = lang_meta.model_dump()

async def _translate_and_update(ctx: TranslationContext) -> None:
    """
    Translate the text in the provided TranslationContext and update the chunk and its metadata.
    
    Performs an async translation via ctx.provider.translate, and when a non-empty, changed translation is returned:
    - replaces ctx.chunk.text with the translated text,
    - attempts to update ctx.chunk.chunk_size (if present),
    - attaches a `translation` entry in ctx.chunk.metadata containing a TranslatedContent dict (original/translated text, source/target languages, provider, and confidence).
    
    If translation fails (exception or None) the original text is preserved and a TranslatedContent record is still attached with confidence 0.0. If the provider returns the same text unchanged, no metadata is attached and the function returns without modifying the chunk.
    
    Parameters:
        ctx (TranslationContext): context carrying provider, chunk, original text, target language, detected language, and content_id.
    
    Returns:
        None
    """
    try:
        tr = await ctx.provider.translate(ctx.text, ctx.target_language)
    except Exception:
        logger.exception("Translation failed for content_id=%s", ctx.content_id)
        tr = None

    translated_text = None
    translation_confidence = 0.0
    provider_used = _provider_name(ctx.provider)
    target_for_meta = _normalize_lang_code(ctx.target_language)

    if tr and isinstance(tr[0], str) and tr[0].strip() and tr[0] != ctx.text:
        translated_text, translation_confidence = tr
        ctx.chunk.text = translated_text
        if hasattr(ctx.chunk, "chunk_size"):
            try:
                ctx.chunk.chunk_size = len(translated_text.split())
            except (AttributeError, ValueError, TypeError):
                logger.debug(
                    "Could not update chunk_size for content_id=%s",
                    ctx.content_id,
                    exc_info=True,
                )
    elif tr is None:
        # Translation failed, keep original text
        translated_text = ctx.text
    else:
        # Provider returned unchanged text
        logger.info("Provider returned unchanged text; skipping translation metadata (content_id=%s)", ctx.content_id)
        return

    trans = TranslatedContent(
        original_chunk_id=str(ctx.content_id),
        original_text=ctx.text,
        translated_text=translated_text,
        source_language=ctx.detected_language,
        target_language=target_for_meta,
        translation_provider=provider_used,
        confidence_score=translation_confidence or 0.0,
    )
    ctx.chunk.metadata["translation"] = trans.model_dump()


# Test helpers for registry isolation
def snapshot_registry() -> Dict[str, Type[TranslationProvider]]:
    """Return a shallow copy snapshot of the provider registry (for tests)."""
    return dict(_provider_registry)

def restore_registry(snapshot: Dict[str, Type[TranslationProvider]]) -> None:
    """
    Restore the global translation provider registry from a previously captured snapshot.
    
    This replaces the current internal provider registry with the given snapshot (clears then updates),
    typically used by tests to restore provider registration state.
    
    Parameters:
        snapshot (Dict[str, Type[TranslationProvider]]): Mapping of provider name keys to provider classes.
    """
    _provider_registry.clear()
    _provider_registry.update(snapshot)

def validate_provider(name: str) -> None:
    """Ensure a provider can be resolved and instantiated or raise."""
    _get_provider(name)

# Built-in Providers
class NoOpProvider:
    """A provider that does nothing, used for testing or disabling translation."""
    async def detect_language(self, _text: str) -> Optional[Tuple[str, float]]:
        """
        No-op language detection: intentionally performs no detection and always returns None.
        
        The `_text` parameter is ignored. Returns None to indicate that this provider does not provide a language detection result.
        """
        return None

    async def translate(self, text: str, _target_language: str) -> Optional[Tuple[str, float]]:
        """
        Return the input text unchanged and a confidence score of 0.0.
        
        This provider does not perform any translation; it mirrors the source text back to the caller.
        Parameters:
            text (str): Source text to "translate".
            _target_language (str): Unused target language parameter.
        Returns:
            Optional[Tuple[str, float]]: A tuple of (text, 0.0).
        """
        return text, 0.0

class LangDetectProvider:
    """
    A provider that uses the 'langdetect' library for offline language detection.
    This provider only detects the language and does not perform translation.
    """
    def __init__(self):
        """
        Initialize the LangDetectProvider by loading the `langdetect.detect_langs` function.
        
        Attempts to import `detect_langs` from the `langdetect` package and stores it on the instance as `_detect_langs`. Raises `LangDetectError` if the `langdetect` dependency is not available.
        """
        try:
            from langdetect import detect_langs  # type: ignore[import-untyped]
            self._detect_langs = detect_langs
        except ImportError as e:
            raise LangDetectError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect the language of `text` using the provider's langdetect backend.
        
        Returns a tuple of (language_code, confidence) where `language_code` is the top
        detected language (e.g., "en") and `confidence` is the detection probability
        in [0.0, 1.0]. Returns None if detection fails or no result is available.
        """
        try:
            detections = await asyncio.to_thread(self._detect_langs, text)
        except Exception:
            logger.exception("Error during language detection")
            return None
        
        if not detections:
            return None
        best_detection = detections[0]
        return best_detection.lang, best_detection.prob

    async def translate(self, text: str, _target_language: str) -> Optional[Tuple[str, float]]:
        # This provider only detects language, does not translate.
        """
        No-op translation: returns the input text unchanged with a 0.0 confidence.
        
        This provider only performs language detection; translate is a passthrough that returns the original `text`
        and a confidence of 0.0 to indicate no translated content was produced.
        
        Returns:
            A tuple of (text, confidence) where `text` is the original input and `confidence` is 0.0.
        """
        return text, 0.0

class OpenAIProvider:
    """A provider that uses OpenAI's API for translation."""
    def __init__(self):
        """
        Initialize the OpenAIProvider by creating an AsyncOpenAI client and loading configuration.
        
        Reads the following environment variables:
        - OPENAI_API_KEY: API key passed to AsyncOpenAI for authentication.
        - OPENAI_TRANSLATE_MODEL: model name to use for translations (default: "gpt-4o-mini").
        - OPENAI_TIMEOUT: request timeout in seconds (default: "30", parsed as float).
        
        Raises:
            OpenAIError: if the OpenAI SDK (AsyncOpenAI) cannot be imported.
        """
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
        """
        Indicates that this provider does not perform standalone language detection.
        
        The OpenAI-based provider does not expose a separate detection endpoint and therefore
        always returns None. Language detection can be achieved by using another provider
        (e.g., the registered langdetect provider) or by incorporating detection into a
        translation prompt if needed.
        """
        return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        """
        Translate the given text to the specified target language using the OpenAI chat completions client.
        
        Parameters:
            text (str): Source text to translate.
            target_language (str): Target language name or code (used verbatim in the translation prompt).
        
        Returns:
            Optional[Tuple[str, float]]: A tuple of (translated_text, confidence). Confidence is 0.0 because no calibrated confidence is available.
            Returns None if translation failed or an error occurred.
        """
        try:
            response = await self.client.with_options(timeout=self.timeout).chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_language}."},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
        except Exception:
            logger.exception("Error during OpenAI translation (model=%s)", self.model)
            return None
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text, 0.0  # No calibrated confidence available.

class GoogleTranslateProvider:
    """A provider that uses the 'googletrans' library for translation."""
    def __init__(self):
        """
        Initialize the GoogleTranslateProvider by importing and instantiating googletrans.Translator.
        
        Raises:
            GoogleTranslateError: If the `googletrans` library is not installed or cannot be imported.
        """
        try:
            from googletrans import Translator  # type: ignore[import-untyped]
            self.translator = Translator()
        except ImportError as e:
            raise GoogleTranslateError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect the language of the given text using the configured googletrans Translator.
        
        Uses a thread to call the synchronous translator.detect method; on failure returns None.
        
        Parameters:
            text: The text to detect the language for.
        
        Returns:
            A tuple (language_code, confidence) where `language_code` is the detected language string from the translator (e.g. "en") and `confidence` is a float in [0.0, 1.0]. Returns None if detection fails.
        """
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
        """
        Translate `text` to `target_language` using the configured googletrans Translator.
        
        Returns a tuple (translated_text, confidence) on success — confidence is always 0.0 because googletrans does not provide a confidence score — or None if translation fails.
        """
        try:
            translation = await asyncio.to_thread(self.translator.translate, text, dest=target_language)
        except Exception:
            logger.exception("Error during Google Translate translation")
            return None
        
        return translation.text, 0.0  # Confidence not provided.

class AzureTranslatorProvider:
    """A provider that uses Azure's Translator service."""
    def __init__(self):
        """
        Initialize the AzureTranslatorProvider.
        
        Attempts to import Azure SDK classes, reads AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_ENDPOINT,
        and AZURE_TRANSLATOR_REGION from the environment, verifies the key is present, and constructs
        a TextTranslationClient using an AzureKeyCredential.
        
        Raises:
            AzureConfigError: if AZURE_TRANSLATOR_KEY is not set.
            AzureTranslateError: if required Azure SDK imports are unavailable.
        """
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
        """
        Detect the language of the given text using the Azure Translator client's detect API.
        
        Attempts to call the Azure client's detect method (using a two-letter region as a country hint when available)
        and returns a tuple of (language_code, confidence_score). Returns None if detection fails or an exception occurs.
        
        Parameters:
            text (str): The text to detect language for.
        
        Returns:
            Optional[Tuple[str, float]]: (ISO language code, confidence between 0.0 and 1.0), or None on error.
        """
        try:
            # Use a valid country hint only when it looks like ISO 3166-1 alpha-2; otherwise omit.
            hint = self.region.lower() if isinstance(self.region, str) and len(self.region) == 2 else None
            response = await asyncio.to_thread(self.client.detect, content=[text], country_hint=hint)
        except Exception:
            logger.exception("Error during Azure language detection")
            return None
        
        detection = response[0].primary_language
        return detection.language, detection.score

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        """
        Translate the given text to the target language using the Azure Translator client.
        
        Parameters:
            text (str): Plain text to translate.
            target_language (str): BCP-47 or ISO language code to translate the text into.
        
        Returns:
            Optional[Tuple[str, float]]: A tuple of (translated_text, confidence). Returns None on error.
            The provider does not surface a numeric confidence score, so the returned confidence is always 0.0.
        """
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
    Translate content chunks to a target language and attach language and translation metadata.
    
    This function accepts either multiple chunk objects as varargs or a single list of chunks.
    For each chunk it:
    - Resolves the named translation provider.
    - Detects the chunk's language (with a fallback detector when available).
    - Decides whether translation is required based on detected language, confidence threshold, and provider.
    - Attaches language metadata (LanguageMetadata) to chunk.metadata.
    - If required, performs translation and updates the chunk text and metadata (TranslatedContent).
    
    Parameters:
        *data_chunks: One or more chunk objects, or a single list of chunk objects. Each chunk must expose a `text` attribute and a `metadata` mapping (the function will create `metadata` if missing).
        target_language (str): Language code to translate into (defaults to TARGET_LANGUAGE).
        translation_provider (str): Registered provider name to use for detection/translation (defaults to "noop").
        confidence_threshold (float): Minimum detection confidence required to skip translation (defaults to CONFIDENCE_THRESHOLD).
    
    Returns:
        list: The list of processed chunk objects (same objects, possibly modified). Metadata keys added include language detection results and, when a translation occurs, translation details.
    """
    provider = _get_provider(translation_provider)
    results = []

    if len(data_chunks) == 1 and isinstance(data_chunks[0], list):
        _chunks = data_chunks[0]
    else:
        _chunks = list(data_chunks)

    for chunk in _chunks:
        ctx = TranslationContext(
            provider=provider,
            chunk=chunk,
            text=getattr(chunk, "text", "") or "",
            target_language=target_language,
            confidence_threshold=confidence_threshold,
            provider_name=translation_provider.lower(),
        )

        ctx.detected_language, ctx.detection_confidence = await _detect_language_with_fallback(
            ctx.provider, ctx.text, str(ctx.content_id)
        )

        _decide_if_translation_is_required(ctx)
        _attach_language_metadata(ctx)

        if ctx.requires_translation:
            await _translate_and_update(ctx)

        results.append(ctx.chunk)

    return results
