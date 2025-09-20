import os
import asyncio
from typing import Optional, Tuple, Any
from ..translation_providers_enum import TranslationProvider
from ..models import TranslationResponse
from cognee.infrastructure.llm.LLMGateway import LLMGateway
import logging

logger = logging.getLogger(__name__)

class LLMProvider:
    """
    A translation provider that uses Cognee's internal LLM abstractions.
    This provider does not support language detection and will rely on a fallback.
    """

    async def detect_language(self, _text: str) -> Optional[Tuple[str, float]]:
        """
        This provider does not support language detection.
        Returns None.
        """
        return None

    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        """
        Translate text using Cognee's internal LLM abstractions.
        Returns a tuple of (translated_text, confidence) or None if translation fails.
        """
        try:
            system_prompt = f"You are a professional translation assistant. Translate the provided text to {target_language}. Provide only the translated text without any additional commentary, quotes, or explanations. Maintain the original meaning and tone as closely as possible."
            translation_response = await LLMGateway.acreate_structured_output(
                text_input=text,
                system_prompt=system_prompt,
                response_model=TranslationResponse
            )
            if translation_response and translation_response.translated_text:
                translated_text = translation_response.translated_text.strip()
                if translated_text and translated_text != text:
                    return translated_text, 0.9  # High confidence for LLM-based translation
        except (ImportError, ValueError) as e:
            logger.error("LLM translation import/value error: %s", e)
        except Exception as e:
            logger.exception("LLM translation failed: %s", str(e))
        return None
