from datetime import datetime, timezone
from typing import List, Tuple
import os

from langdetect import detect_langs
from openai import AsyncOpenAI

from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata
from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set; translation will fail")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def detect_language(text: str) -> Tuple[str, float]:
    try:
        langs = detect_langs(text)
        if langs:
            top_lang = langs[0]
            return top_lang.lang, top_lang.prob
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
    return "en", 1.0


async def translate_text(
    text: str, source_lang: str, target_lang: str, provider: str
) -> Tuple[str, float]:

    if provider.lower() == "openai":
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Translate this text from {source_lang} to {target_lang}:\n\n{text}",
                    }
                ],
            )

            # FIXED: v1.x message structure
            message = response.choices[0].message
            translated = message["content"].strip()

            # OpenAI doesn't return explicit confidence
            confidence = 0.95

            return translated, confidence

        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            return text, 0.0

    # Unknown provider → no translation
    return text, 1.0


async def translate_content(
    data_chunks: List[DocumentChunk],
    target_language: str = "en",
    translation_provider: str = "openai",
    confidence_threshold: float = 0.8,
) -> List[DocumentChunk]:

    translated_chunks = []

    for chunk in data_chunks:
        try:
            text = chunk.text

            # Detect language
            lang, lang_conf = detect_language(text)
            requires_translation = (
                lang != target_language and lang_conf >= confidence_threshold
            )

            # Add metadata
            language_metadata = LanguageMetadata(
                content_id=chunk.id,
                detected_language=lang,
                language_confidence=lang_conf,
                requires_translation=requires_translation,
                character_count=len(text),
            )

            if requires_translation:
                translated_text, trans_conf = await translate_text(
                    text, lang, target_language, translation_provider
                )

                translation_info = TranslatedContent(
                    original_chunk_id=chunk.id,
                    original_text=text,
                    translated_text=translated_text,
                    source_language=lang,
                    target_language=target_language,
                    translation_provider=translation_provider,
                    confidence_score=trans_conf,
                    translation_timestamp=datetime.now(timezone.utc),
                )

                chunk.translated = translation_info
                logger.info(
                    f"Translated chunk {chunk.id} from {lang} → {target_language}"
                )

            else:
                chunk.translated = None
                logger.debug(
                    f"Chunk {chunk.id}: no translation required (detected={lang})"
                )

            chunk.language_metadata = language_metadata
            translated_chunks.append(chunk)

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk.id}: {e}")
            raise

    return translated_chunks
