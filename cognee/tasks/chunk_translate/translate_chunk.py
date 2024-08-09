
import logging

from cognee.base_config import get_base_config

BaseConfig = get_base_config()

async def translate_text(data, source_language:str='sr', target_language:str='en', region_name='eu-west-1'):
    """
    Translate text from source language to target language using AWS Translate.
    Parameters:
    data (str): The text to be translated.
    source_language (str): The source language code (e.g., 'sr' for Serbian). ISO 639-2 Code https://www.loc.gov/standards/iso639-2/php/code_list.php
    target_language (str): The target language code (e.g., 'en' for English). ISO 639-2 Code https://www.loc.gov/standards/iso639-2/php/code_list.php
    region_name (str): AWS region name.
    Returns:
    str: Translated text or an error message.
    """
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    if not data:
        yield "No text provided for translation."

    if not source_language or not target_language:
        yield "Both source and target language codes are required."

    try:
        translate = boto3.client(service_name='translate', region_name=region_name, use_ssl=True)
        result = translate.translate_text(Text=data, SourceLanguageCode=source_language, TargetLanguageCode=target_language)
        yield result.get('TranslatedText', 'No translation found.')

    except BotoCoreError as e:
        logging.info(f"BotoCoreError occurred: {e}")
        yield "Error with AWS Translate service configuration or request."

    except ClientError as e:
        logging.info(f"ClientError occurred: {e}")
        yield "Error with AWS client or network issue."