import os
import asyncio
from typing import Optional, Tuple, Any
from ..translation_providers_enum import TranslationProvider
from ..translation_errors import AzureTranslateError, AzureConfigError
import logging

logger = logging.getLogger(__name__)

class AzureTranslateProvider:
    """
    A translation provider that uses Azure's Text Translation API.
    This provider supports both language detection and translation.
    """
    _client: Any = None

    def __init__(self):
        cls = type(self)
        if cls._client is None:
            try:
                from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
                from azure.core.credentials import AzureKeyCredential

                key = os.getenv("AZURE_TRANSLATE_KEY")
                endpoint = os.getenv("AZURE_TRANSLATE_ENDPOINT")
                region = os.getenv("AZURE_TRANSLATE_REGION")

                if not key:
                    raise AzureConfigError()
                if not endpoint:
                    endpoint = "https://api.cognitive.microsofttranslator.com"
                if region:
                    cred = TranslatorCredential(key, region)
                    cls._client = TextTranslationClient(endpoint=endpoint, credential=cred)
                else:
                    cred = AzureKeyCredential(key)
                    cls._client = TextTranslationClient(endpoint=endpoint, credential=cred)
            except ImportError as e:
                raise AzureTranslateError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            timeout = float(os.getenv("AZURE_TRANSLATE_TIMEOUT", "30"))
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(type(self)._client.detect, content=[text]),
                    timeout=timeout
                )
            except TypeError:
                response = await asyncio.wait_for(
                    asyncio.to_thread(type(self)._client.detect, [text]),
                    timeout=timeout
                )
        except (ValueError, AttributeError, TypeError, asyncio.TimeoutError):
            logger.exception("Azure Translate language detection failed")
            return None
        except (ImportError, RuntimeError):
            logger.exception("Azure Translate language detection failed (SDK error)")
            return None
        if response and getattr(response[0], "detected_language", None):
            dl = response[0].detected_language
            return dl.language, dl.score
        return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            timeout = float(os.getenv("AZURE_TRANSLATE_TIMEOUT", "30"))
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(type(self)._client.translate, content=[text], to=[target_language]),
                    timeout=timeout
                )
            except TypeError:
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(type(self)._client.translate, [text], to_language=[target_language]),
                        timeout=timeout
                    )
                except TypeError:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            type(self)._client.translate, body=[text], to_language=[target_language]
                        ),
                        timeout=timeout
                    )
        except (ValueError, AttributeError, TypeError, asyncio.TimeoutError):
            logger.exception("Azure Translate translation failed")
            return None
        except (ImportError, RuntimeError):
            logger.exception("Azure Translate translation failed (SDK error)")
            return None
        if response and response[0].translations:
            return response[0].translations[0].text, 0.85  # High confidence for Azure Translate
        return None
