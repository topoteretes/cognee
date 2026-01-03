from dataclasses import dataclass
from typing import Optional

from cognee.shared.logging_utils import get_logger

from .config import get_translation_config
from .exceptions import LanguageDetectionError

logger = get_logger(__name__)


# ISO 639-1 language code to name mapping
LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""

    language_code: str
    language_name: str
    confidence: float
    requires_translation: bool
    character_count: int


def get_language_name(language_code: str) -> str:
    """Get the human-readable name for a language code."""
    return LANGUAGE_NAMES.get(language_code.lower(), language_code)


def detect_language(
    text: str,
    target_language: str = "en",
    confidence_threshold: Optional[float] = None,
) -> LanguageDetectionResult:
    """
    Detect the language of the given text.

    Uses the langdetect library which is already a dependency of cognee.

    Args:
        text: The text to analyze
        target_language: The target language for translation comparison
        confidence_threshold: Minimum confidence to consider detection reliable

    Returns:
        LanguageDetectionResult with language info and translation requirement

    Raises:
        LanguageDetectionError: If language detection fails
    """
    config = get_translation_config()
    threshold = confidence_threshold or config.confidence_threshold

    # Handle empty or very short text
    if not text or len(text.strip()) < config.min_text_length_for_detection:
        if config.skip_detection_for_short_text:
            return LanguageDetectionResult(
                language_code="unknown",
                language_name="Unknown",
                confidence=0.0,
                requires_translation=False,
                character_count=len(text) if text else 0,
            )
        else:
            raise LanguageDetectionError(
                f"Text too short for reliable language detection: {len(text)} characters"
            )

    try:
        from langdetect import detect_langs, LangDetectException
    except ImportError:
        raise LanguageDetectionError(
            "langdetect is required for language detection. Install it with: pip install langdetect"
        )

    try:
        # Get detection results with probabilities
        detections = detect_langs(text)

        if not detections:
            raise LanguageDetectionError("No language detected")

        # Get the most likely language
        best_detection = detections[0]
        language_code = best_detection.lang
        confidence = best_detection.prob

        # Check if translation is needed
        requires_translation = (
            language_code.lower() != target_language.lower() and confidence >= threshold
        )

        return LanguageDetectionResult(
            language_code=language_code,
            language_name=get_language_name(language_code),
            confidence=confidence,
            requires_translation=requires_translation,
            character_count=len(text),
        )

    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        raise LanguageDetectionError(f"Language detection failed: {e}", original_error=e)
    except Exception as e:
        logger.error(f"Unexpected error during language detection: {e}")
        raise LanguageDetectionError(
            f"Unexpected error during language detection: {e}", original_error=e
        )


async def detect_language_async(
    text: str,
    target_language: str = "en",
    confidence_threshold: Optional[float] = None,
) -> LanguageDetectionResult:
    """
    Async wrapper for language detection.

    Args:
        text: The text to analyze
        target_language: The target language for translation comparison
        confidence_threshold: Minimum confidence to consider detection reliable

    Returns:
        LanguageDetectionResult with language info and translation requirement
    """
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, detect_language, text, target_language, confidence_threshold
    )
