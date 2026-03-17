import asyncio
from typing import Optional

from cognee.shared.logging_utils import get_logger

from .base import TranslationProvider, TranslationResult
from ..config import get_translation_config

logger = get_logger(__name__)


class GoogleTranslationProvider(TranslationProvider):
    """
    Translation provider using Google Cloud Translation API.

    Requires:
    - google-cloud-translate package
    - GOOGLE_TRANSLATE_API_KEY or GOOGLE_PROJECT_ID environment variable
    """

    def __init__(self):
        self._client = None
        self._config = get_translation_config()

    @property
    def provider_name(self) -> str:
        return "google"

    def _get_client(self):
        """Lazy initialization of Google Translate client."""
        if self._client is None:
            try:
                from google.cloud import translate_v2 as translate

                self._client = translate.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-translate is required for Google translation. "
                    "Install it with: pip install google-cloud-translate"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Google Translate client: {e}")
                raise
        return self._client

    def is_available(self) -> bool:
        """Check if Google Translate is available."""
        try:
            self._get_client()
            return True
        except Exception as e:
            logger.debug(f"Google Translate not available: {e}")
            return False

    async def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text using Google Translate API.

        Args:
            text: The text to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (optional)

        Returns:
            TranslationResult with translated text and metadata
        """
        try:
            client = self._get_client()

            # Run in thread pool since google-cloud-translate is synchronous
            loop = asyncio.get_running_loop()

            # Build kwargs for translate call
            translate_kwargs = {"target_language": target_language}
            if source_language:
                translate_kwargs["source_language"] = source_language

            result = await loop.run_in_executor(
                None,
                lambda: client.translate(text, **translate_kwargs),
            )

            detected_language = result.get("detectedSourceLanguage", source_language or "unknown")

            return TranslationResult(
                translated_text=result["translatedText"],
                source_language=detected_language,
                target_language=target_language,
                # Google Translate API does not provide confidence scores
                confidence_score=None,
                provider=self.provider_name,
                raw_response=result,
            )

        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            raise

    async def translate_batch(
        self,
        texts: list[str],
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts using Google Translate API.

        Google Translate supports batch translation natively.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (optional)

        Returns:
            List of TranslationResult objects
        """
        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()

            # Build kwargs for translate call
            translate_kwargs = {"target_language": target_language}
            if source_language:
                translate_kwargs["source_language"] = source_language

            results = await loop.run_in_executor(
                None,
                lambda: client.translate(texts, **translate_kwargs),
            )

            translation_results = []
            for result in results:
                detected_language = result.get(
                    "detectedSourceLanguage", source_language or "unknown"
                )
                translation_results.append(
                    TranslationResult(
                        translated_text=result["translatedText"],
                        source_language=detected_language,
                        target_language=target_language,
                        # Google Translate API does not provide confidence scores
                        confidence_score=None,
                        provider=self.provider_name,
                        raw_response=result,
                    )
                )

            return translation_results

        except Exception as e:
            logger.error(f"Google batch translation failed: {e}")
            raise
