
import logging



async def detect_language(data:str):
    """
    Detect the language of the given text and return its ISO 639-1 language code.
    If the detected language is Croatian ('hr'), it maps to Serbian ('sr').
    The text is trimmed to the first 100 characters for efficient processing.
    Parameters:
    text (str): The text for language detection.
    Returns:
    str: The ISO 639-1 language code of the detected language, or 'None' in case of an error.
    """

    # Trim the text to the first 100 characters
    from langdetect import detect, LangDetectException
    trimmed_text = data[:100]

    try:
        # Detect the language using langdetect
        detected_lang_iso639_1 = detect(trimmed_text)
        logging.info(f"Detected ISO 639-1 code: {detected_lang_iso639_1}")

        # Special case: map 'hr' (Croatian) to 'sr' (Serbian ISO 639-2)
        if detected_lang_iso639_1 == 'hr':
            yield 'sr'
        yield detected_lang_iso639_1

    except LangDetectException as e:
        logging.error(f"Language detection error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    yield None