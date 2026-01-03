import asyncio
from typing import List, Optional
from uuid import uuid5

from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.logging_utils import get_logger

from .config import get_translation_config, TranslationProviderType
from .detect_language import detect_language_async, LanguageDetectionResult
from .exceptions import TranslationError, LanguageDetectionError
from .models import TranslatedContent, LanguageMetadata
from .providers import get_translation_provider, TranslationResult

logger = get_logger(__name__)


async def translate_content(
    data_chunks: List[DocumentChunk],
    target_language: str = None,
    translation_provider: TranslationProviderType = None,
    confidence_threshold: float = None,
    skip_if_target_language: bool = True,
    preserve_original: bool = True,
) -> List[DocumentChunk]:
    """
    Translate non-English content to the target language.

    This task detects the language of each document chunk and translates
    non-target-language content using the specified translation provider.
    Original text is preserved alongside translated versions.

    Args:
        data_chunks: List of DocumentChunk objects to process
        target_language: Target language code (default: "en" for English)
                        If not provided, uses config default
        translation_provider: Translation service to use ("llm", "google", "azure")
                            If not provided, uses config default
        confidence_threshold: Minimum confidence for language detection (0.0 to 1.0)
                            If not provided, uses config default
        skip_if_target_language: If True, skip chunks already in target language
        preserve_original: If True, store original text in TranslatedContent

    Returns:
        List of DocumentChunk objects with translated content.
        Chunks that required translation will have TranslatedContent
        objects in their 'contains' list.

    Note:
        This function mutates the input chunks in-place. Specifically:
        - chunk.text is replaced with the translated text
        - chunk.contains is updated with LanguageMetadata and TranslatedContent
        The original text is preserved in TranslatedContent.original_text
        if preserve_original=True.

    Example:
        ```python
        from cognee.tasks.translation import translate_content

        # Translate chunks using default settings
        translated_chunks = await translate_content(chunks)

        # Translate with specific provider
        translated_chunks = await translate_content(
            chunks,
            translation_provider="llm",
            confidence_threshold=0.9
        )
        ```
    """
    if not isinstance(data_chunks, list):
        raise TranslationError("data_chunks must be a list")

    if len(data_chunks) == 0:
        return data_chunks

    # Get configuration
    config = get_translation_config()
    provider_name = translation_provider or config.translation_provider
    target_lang = target_language or config.target_language
    threshold = confidence_threshold or config.confidence_threshold

    logger.info(
        f"Starting translation task for {len(data_chunks)} chunks "
        f"using {provider_name} provider, target language: {target_lang}"
    )

    # Get the translation provider
    provider = get_translation_provider(provider_name)

    # Process chunks
    processed_chunks = []
    total_chunks = len(data_chunks)

    for chunk_index, chunk in enumerate(data_chunks):
        # Log progress for large batches
        if chunk_index > 0 and chunk_index % 100 == 0:
            logger.info(f"Translation progress: {chunk_index}/{total_chunks} chunks processed")

        if not hasattr(chunk, "text") or not chunk.text:
            processed_chunks.append(chunk)
            continue

        try:
            # Detect language
            detection = await detect_language_async(chunk.text, target_lang, threshold)

            # Create language metadata
            language_metadata = LanguageMetadata(
                id=uuid5(chunk.id, "LanguageMetadata"),
                content_id=chunk.id,
                detected_language=detection.language_code,
                language_confidence=detection.confidence,
                requires_translation=detection.requires_translation,
                character_count=detection.character_count,
                language_name=detection.language_name,
            )

            # Skip if already in target language
            if not detection.requires_translation:
                if skip_if_target_language:
                    logger.debug(
                        f"Skipping chunk {chunk.id}: already in target language "
                        f"({detection.language_code})"
                    )
                    # Add language metadata to chunk
                    _add_to_chunk_contains(chunk, language_metadata)
                    processed_chunks.append(chunk)
                    continue

            # Translate the content
            logger.debug(
                f"Translating chunk {chunk.id} from {detection.language_code} to {target_lang}"
            )

            translation_result = await provider.translate(
                text=chunk.text,
                target_language=target_lang,
                source_language=detection.language_code,
            )

            # Create TranslatedContent data point
            translated_content = TranslatedContent(
                id=uuid5(chunk.id, "TranslatedContent"),
                original_chunk_id=chunk.id,
                original_text=chunk.text if preserve_original else "",
                translated_text=translation_result.translated_text,
                source_language=translation_result.source_language,
                target_language=translation_result.target_language,
                translation_provider=translation_result.provider,
                confidence_score=translation_result.confidence_score,
                translated_from=chunk,
            )

            # Update chunk text with translated content
            chunk.text = translation_result.translated_text

            # Add metadata to chunk's contains list
            _add_to_chunk_contains(chunk, language_metadata)
            _add_to_chunk_contains(chunk, translated_content)

            processed_chunks.append(chunk)

            logger.debug(
                f"Successfully translated chunk {chunk.id}: "
                f"{detection.language_code} -> {target_lang}"
            )

        except LanguageDetectionError as e:
            logger.warning(f"Language detection failed for chunk {chunk.id}: {e}")
            processed_chunks.append(chunk)
        except TranslationError as e:
            logger.error(f"Translation failed for chunk {chunk.id}: {e}")
            processed_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Unexpected error processing chunk {chunk.id}: {e}")
            processed_chunks.append(chunk)

    logger.info(f"Translation task completed for {len(processed_chunks)} chunks")
    return processed_chunks


def _add_to_chunk_contains(chunk: DocumentChunk, item) -> None:
    """Helper to add an item to a chunk's contains list."""
    if chunk.contains is None:
        chunk.contains = []
    chunk.contains.append(item)


async def translate_text(
    text: str,
    target_language: str = None,
    translation_provider: TranslationProviderType = None,
    source_language: Optional[str] = None,
) -> TranslationResult:
    """
    Translate a single text string.

    This is a convenience function for translating individual texts
    without creating DocumentChunk objects.

    Args:
        text: The text to translate
        target_language: Target language code (default: uses config, typically "en")
                        If not provided, uses config default
        translation_provider: Translation service to use
                            If not provided, uses config default
        source_language: Source language code (optional, auto-detected if not provided)

    Returns:
        TranslationResult with translated text and metadata

    Example:
        ```python
        from cognee.tasks.translation import translate_text

        result = await translate_text(
            "Bonjour le monde!",
            target_language="en"
        )
        print(result.translated_text)  # "Hello world!"
        print(result.source_language)  # "fr"
        ```
    """
    config = get_translation_config()
    provider_name = translation_provider or config.translation_provider
    target_lang = target_language or config.target_language

    provider = get_translation_provider(provider_name)

    return await provider.translate(
        text=text,
        target_language=target_lang,
        source_language=source_language,
    )


async def batch_translate_texts(
    texts: List[str],
    target_language: str = None,
    translation_provider: TranslationProviderType = None,
    source_language: Optional[str] = None,
) -> List[TranslationResult]:
    """
    Translate multiple text strings in batch.

    This is more efficient than translating texts individually,
    especially for providers that support native batch operations.

    Args:
        texts: List of texts to translate
        target_language: Target language code (default: uses config, typically "en")
                        If not provided, uses config default
        translation_provider: Translation service to use
                            If not provided, uses config default
        source_language: Source language code (optional)

    Returns:
        List of TranslationResult objects

    Example:
        ```python
        from cognee.tasks.translation import batch_translate_texts

        results = await batch_translate_texts(
            ["Hola", "¿Cómo estás?", "Adiós"],
            target_language="en"
        )
        for result in results:
            print(f"{result.source_language}: {result.translated_text}")
        ```
    """
    config = get_translation_config()
    provider_name = translation_provider or config.translation_provider
    target_lang = target_language or config.target_language

    provider = get_translation_provider(provider_name)

    return await provider.translate_batch(
        texts=texts,
        target_language=target_lang,
        source_language=source_language,
    )
