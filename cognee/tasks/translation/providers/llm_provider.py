import asyncio
from typing import Optional

from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.shared.logging_utils import get_logger

from .base import TranslationProvider, TranslationResult

logger = get_logger(__name__)


class TranslationOutput(BaseModel):
    """Pydantic model for structured translation output from LLM."""

    translated_text: str
    detected_source_language: str
    translation_notes: Optional[str] = None


class LLMTranslationProvider(TranslationProvider):
    """
    Translation provider using the configured LLM for translation.

    This provider leverages the existing LLM infrastructure in Cognee
    to perform translations using any LLM configured via LLM_PROVIDER
    (OpenAI, Azure, Ollama, Anthropic, etc.).

    The LLM used is determined by the cognee LLM configuration settings:
    - LLM_PROVIDER: The LLM provider (openai, azure, ollama, etc.)
    - LLM_MODEL: The model to use
    - LLM_API_KEY: API key for the provider
    """

    @property
    def provider_name(self) -> str:
        """Return 'llm' as the provider name."""
        return "llm"

    async def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text using the configured LLM.

        Args:
            text: The text to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (optional)

        Returns:
            TranslationResult with translated text and metadata
        """
        try:
            system_prompt = read_query_prompt("translate_content.txt")

            # Validate system prompt was loaded successfully
            if system_prompt is None:
                logger.warning("translate_content.txt prompt file not found, using default prompt")
                system_prompt = (
                    "You are a professional translator. Translate the given text accurately "
                    "while preserving the original meaning, tone, and style. "
                    "Detect the source language if not provided."
                )

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
                # TODO: Consider deriving confidence from LLM response metadata
                # or making configurable via TranslationConfig
                confidence_score=0.95,  # LLM translations are generally high quality
                provider=self.provider_name,
                raw_response={"notes": result.translation_notes},
            )

        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            raise

    async def translate_batch(
        self,
        texts: list[str],
        target_language: str = "en",
        source_language: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts using the configured LLM.

        Uses a semaphore to limit concurrent requests and avoid API rate limits.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (optional)
            max_concurrent: Maximum concurrent translation requests (default: 5)

        Returns:
            List of TranslationResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_translate(text: str) -> TranslationResult:
            async with semaphore:
                return await self.translate(text, target_language, source_language)

        tasks = [limited_translate(text) for text in texts]
        return await asyncio.gather(*tasks)

    def is_available(self) -> bool:
        """Check if LLM provider is available (has required credentials)."""
        try:
            llm_config = get_llm_config()
            # Check if API key is configured (required for most providers)
            return bool(llm_config.llm_api_key)
        except Exception:
            return False
