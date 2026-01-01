import asyncio
from typing import Optional

from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.shared.logging_utils import get_logger

from .base import TranslationProvider, TranslationResult

logger = get_logger(__name__)


class TranslationOutput(BaseModel):
    """Pydantic model for structured translation output from LLM."""

    translated_text: str
    detected_source_language: str
    translation_notes: Optional[str] = None


class OpenAITranslationProvider(TranslationProvider):
    """
    Translation provider using OpenAI's LLM for translation.

    This provider leverages the existing LLM infrastructure in Cognee
    to perform translations using GPT models.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    async def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text using OpenAI's LLM.

        Args:
            text: The text to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (optional)

        Returns:
            TranslationResult with translated text and metadata
        """
        try:
            system_prompt = read_query_prompt("translate_content.txt")

            # Build the input with context
            if source_language:
                input_text = (
                    f"Translate the following text from {source_language} to {target_language}.\n\n"
                    f"Text to translate:\n{text}"
                )
            else:
                input_text = (
                    f"Translate the following text to {target_language}. "
                    f"First detect the source language.\n\n"
                    f"Text to translate:\n{text}"
                )

            result = await LLMGateway.acreate_structured_output(
                text_input=input_text,
                system_prompt=system_prompt,
                response_model=TranslationOutput,
            )

            return TranslationResult(
                translated_text=result.translated_text,
                source_language=source_language or result.detected_source_language,
                target_language=target_language,
                confidence_score=0.95,  # LLM translations are generally high quality
                provider=self.provider_name,
                raw_response={"notes": result.translation_notes},
            )

        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            raise

    async def translate_batch(
        self,
        texts: list[str],
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts using OpenAI's LLM.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (optional)

        Returns:
            List of TranslationResult objects
        """
        tasks = [
            self.translate(text, target_language, source_language) for text in texts
        ]
        return await asyncio.gather(*tasks)
