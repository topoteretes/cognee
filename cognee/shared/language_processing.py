""" This module provides language processing functions for language detection and translation. """
import logging
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langdetect import detect, LangDetectException
import iso639


# Basic configuration of the logging system
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def detect_language(text):
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
    trimmed_text = text[:100]

    try:
        # Detect the language using langdetect
        detected_lang_iso639_1 = detect(trimmed_text)
        logging.info(f"Detected ISO 639-1 code: %s {detected_lang_iso639_1}")

        # Special case: map 'hr' (Croatian) to 'sr' (Serbian ISO 639-2)
        if detected_lang_iso639_1 == "hr":
            return "sr"
        return detected_lang_iso639_1

    except LangDetectException as e:
        logging.error(f"Language detection error: %s  {e}")
    except Exception as e:
        logging.error(f"Unexpected error: %s  {e}")

    return -1


def translate_text(
    text,
    source_language: str = "sr",
    target_language: str = "en",
    region_name="eu-west-1",
):
    """
    Translate text from source language to target language using AWS Translate.


    Parameters:
    text (str): The text to be translated.
    source_language (str): The source language code (e.g., 'sr' for Serbian).
     ISO 639-2 Code https://www.loc.gov/standards/iso639-2/php/code_list.php
    target_language (str): The target language code (e.g., 'en' for English).
    ISO 639-2 Code https://www.loc.gov/standards/iso639-2/php/code_list.php
    region_name (str): AWS region name.

    Returns:
    str: Translated text or an error message.
    """
    if not text:
        return "No text provided for translation."

    if not source_language or not target_language:
        return "Both source and target language codes are required."

    try:
        translate = boto3.client(
            service_name="translate", region_name=region_name, use_ssl=True
        )
        result = translate.translate_text(
            Text=text,
            SourceLanguageCode=source_language,
            TargetLanguageCode=target_language,
        )
        return result.get("TranslatedText", "No translation found.")

    except BotoCoreError as e:
        logging.info(f"BotoCoreError occurred: %s  {e}")
        return "Error with AWS Translate service configuration or request."

    except ClientError as e:
        logging.info(f"ClientError occurred: %s  {e}")
        return "Error with AWS client or network issue."
