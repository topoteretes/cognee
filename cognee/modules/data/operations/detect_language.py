from cognee.shared.logging_utils import get_logger, ERROR

logger = get_logger(level=ERROR)


async def detect_language(text: str):
    """
    Detect the language of the given text and return its ISO 639-1 language code.
    If the detected language is Croatian ("hr"), it maps to Serbian ("sr").
    The text is trimmed to the first 100 characters for efficient processing.
    Parameters:
    text (str): The text for language detection.
    Returns:
    str: The ISO 639-1 language code of the detected language, or "None" in case of an error.
    """

    from langdetect import detect, LangDetectException

    # Trim the text to the first 100 characters
    trimmed_text = text[:100]

    try:
        # Detect the language using langdetect
        detected_lang_iso639_1 = detect(trimmed_text)

        # Special case: map "hr" (Croatian) to "sr" (Serbian ISO 639-2)
        if detected_lang_iso639_1 == "hr":
            return "sr"

        return detected_lang_iso639_1

    except LangDetectException as e:
        logger.error(f"Language detection error: {e}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e

    return None
