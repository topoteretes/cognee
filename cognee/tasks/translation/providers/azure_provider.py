from typing import Optional

import aiohttp

from cognee.shared.logging_utils import get_logger

from .base import TranslationProvider, TranslationResult
from ..config import get_translation_config
from ..exceptions import TranslationProviderError

logger = get_logger(__name__)


class AzureTranslationProvider(TranslationProvider):
    """
    Translation provider using Azure Translator API.

    Requires:
    - AZURE_TRANSLATOR_KEY environment variable
    - AZURE_TRANSLATOR_REGION environment variable (optional)
    """

    def __init__(self):
        self._config = get_translation_config()

    @property
    def provider_name(self) -> str:
        return "azure"

    def is_available(self) -> bool:
        """Check if Azure Translator is available."""
        return self._config.azure_translator_key is not None

    async def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text using Azure Translator API.

        Args:
            text: The text to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (optional)

        Returns:
            TranslationResult with translated text and metadata
        """
        if not self.is_available():
            raise TranslationProviderError(
                provider=self.provider_name,
                message="Azure Translator API key not configured. Set AZURE_TRANSLATOR_KEY environment variable.",
            )

        endpoint = f"{self._config.azure_translator_endpoint}/translate"

        params = {
            "api-version": "3.0",
            "to": target_language,
        }
        if source_language:
            params["from"] = source_language

        headers = {
            "Ocp-Apim-Subscription-Key": self._config.azure_translator_key,
            "Content-Type": "application/json",
        }
        if self._config.azure_translator_region:
            headers["Ocp-Apim-Subscription-Region"] = self._config.azure_translator_region

        body = [{"text": text}]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self._config.timeout_seconds),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            translation = result[0]["translations"][0]
            detected_language = result[0].get("detectedLanguage", {})

            return TranslationResult(
                translated_text=translation["text"],
                source_language=source_language or detected_language.get("language", "unknown"),
                target_language=target_language,
                confidence_score=detected_language.get("score", 0.9),
                provider=self.provider_name,
                raw_response=result[0],
            )

        except Exception as e:
            logger.error(f"Azure translation failed: {e}")
            raise TranslationProviderError(
                provider=self.provider_name,
                message=f"Translation failed: {e}",
                original_error=e,
            )

    async def translate_batch(
        self,
        texts: list[str],
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts using Azure Translator API.

        Azure Translator supports up to 100 texts per request.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (optional)

        Returns:
            List of TranslationResult objects
        """
        if not self.is_available():
            raise TranslationProviderError(
                provider=self.provider_name,
                message="Azure Translator API key not configured. Set AZURE_TRANSLATOR_KEY environment variable.",
            )

        endpoint = f"{self._config.azure_translator_endpoint}/translate"

        params = {
            "api-version": "3.0",
            "to": target_language,
        }
        if source_language:
            params["from"] = source_language

        headers = {
            "Ocp-Apim-Subscription-Key": self._config.azure_translator_key,
            "Content-Type": "application/json",
        }
        if self._config.azure_translator_region:
            headers["Ocp-Apim-Subscription-Region"] = self._config.azure_translator_region

        # Azure supports up to 100 texts per request
        batch_size = min(100, self._config.batch_size)
        all_results = []

        try:
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    body = [{"text": text} for text in batch]

                    async with session.post(
                        endpoint,
                        params=params,
                        headers=headers,
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=self._config.timeout_seconds),
                    ) as response:
                        response.raise_for_status()
                        results = await response.json()

                    for result in results:
                        translation = result["translations"][0]
                        detected_language = result.get("detectedLanguage", {})

                        all_results.append(
                            TranslationResult(
                                translated_text=translation["text"],
                                source_language=source_language
                                or detected_language.get("language", "unknown"),
                                target_language=target_language,
                                confidence_score=detected_language.get("score", 0.9),
                                provider=self.provider_name,
                                raw_response=result,
                            )
                        )

        except Exception as e:
            logger.error(f"Azure batch translation failed: {e}")
            raise TranslationProviderError(
                provider=self.provider_name,
                message=f"Batch translation failed: {e}",
                original_error=e,
            )

        return all_results
