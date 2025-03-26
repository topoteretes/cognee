from cognee.shared.logging_utils import get_logger, ERROR

from cognee.exceptions import InvalidValueError

logger = get_logger(level=ERROR)


async def translate_text(
    text, source_language: str = "sr", target_language: str = "en", region_name="eu-west-1"
):
    """
    Translate text from source language to target language using AWS Translate.
    Parameters:
    text (str): The text to be translated.
    source_language (str): The source language code (e.g., "sr" for Serbian). ISO 639-2 Code https://www.loc.gov/standards/iso639-2/php/code_list.php
    target_language (str): The target language code (e.g., "en" for English). ISO 639-2 Code https://www.loc.gov/standards/iso639-2/php/code_list.php
    region_name (str): AWS region name.
    Returns:
    str: Translated text or an error message.
    """

    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    if not text:
        raise InvalidValueError(message="No text to translate.")

    if not source_language or not target_language:
        raise InvalidValueError(message="Source and target language codes are required.")

    try:
        translate = boto3.client(service_name="translate", region_name=region_name, use_ssl=True)
        result = translate.translate_text(
            Text=text,
            SourceLanguageCode=source_language,
            TargetLanguageCode=target_language,
        )
        yield result.get("TranslatedText", "No translation found.")

    except BotoCoreError as e:
        logger.error(f"BotoCoreError occurred: {e}")
        yield None

    except ClientError as e:
        logger.error(f"ClientError occurred: {e}")
        yield None
