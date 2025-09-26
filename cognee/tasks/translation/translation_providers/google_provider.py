import os
import asyncio
from typing import Optional, Tuple, Any
from ..translation_providers_enum import TranslationProvider
from ..translation_errors import GoogleTranslateError
import logging

logger = logging.getLogger(__name__)

class GoogleTranslateProvider:
    """
    A translation provider that uses the 'googletrans' library.
    This provider supports both language detection and translation.
    """
    _translator: Any = None

    def __init__(self):
        cls = type(self)
        if cls._translator is None:
            try:
                from googletrans import Translator
                cls._translator = Translator()
            except ImportError as e:
                raise GoogleTranslateError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            timeout = float(os.getenv("GOOGLE_TRANSLATE_TIMEOUT", "30"))
            detection = await asyncio.wait_for(
                asyncio.to_thread(type(self)._translator.detect, text), 
                timeout=timeout
            )
        except (AttributeError, TypeError, ValueError, asyncio.TimeoutError):
            logger.exception("Google Translate language detection failed")
            return None
        try:
            confidence = getattr(detection, "confidence", 0.0) or 0.0
        except (TypeError, ValueError, AttributeError):
            confidence = 0.0
        return detection.lang, confidence

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        try:
            timeout = float(os.getenv("GOOGLE_TRANSLATE_TIMEOUT", "30"))
            translation = await asyncio.wait_for(
                asyncio.to_thread(type(self)._translator.translate, text, dest=target_language),
                timeout=timeout
            )
        except (AttributeError, TypeError, ValueError, asyncio.TimeoutError):
            logger.exception("Google Translate translation failed")
            return None
        return translation.text, 0.8  # Moderate confidence for Google Translate
