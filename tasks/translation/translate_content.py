from typing import List
from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata
# Import DocumentChunk from the appropriate module

async def translate_content(
    data_chunks: List['DocumentChunk'],
    target_language: str = "en",
    translation_provider: str = "openai",
    confidence_threshold: float = 0.8
) -> List['DocumentChunk']:
    """
    Translate non-English content to target language

    Args:
        data_chunks: Document chunks to process
        target_language: Target language code (default: "en")
        translation_provider: Translation service to use
        confidence_threshold: Minimum confidence for language detection

    Returns:
        Enhanced chunks with translated content and metadata
    """
    # TODO: Implement translation logic here
    from __future__ import annotations

    from typing import List, Protocol, Tuple, Optional
    import logging

    from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata

    # Importing DocumentChunk from the project's chunking models
    from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

    logger = logging.getLogger(__name__)


    class TranslationProvider(Protocol):
        async def detect_language(self, text: str) -> Tuple[str, float]:
            ...

        async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
            ...


    class NoOpProvider:
        """A provider that performs no translation and returns 'en' if text is ASCII."""

        async def detect_language(self, text: str) -> Tuple[str, float]:
            # Simple heuristic: if non-ascii characters exist, mark unknown
            try:
                text.encode("ascii")
                return "en", 0.9
            except UnicodeEncodeError:
                return "unknown", 0.4

        async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
            # No translation performed
            return text, 1.0


    try:
        # optional import: langdetect provides a good local detection fallback
        from langdetect import detect_langs


        class LangDetectProvider(NoOpProvider):
            async def detect_language(self, text: str) -> Tuple[str, float]:
                try:
                    langs = detect_langs(text)
                    if not langs:
                        return "unknown", 0.0
                    top = langs[0]
                    return top.lang, float(top.prob)
                except Exception:
                    return await super().detect_language(text)

    except Exception:
        LangDetectProvider = NoOpProvider


    try:
        import os
        import openai


        class OpenAIProvider:
            def __init__(self):
                key = os.getenv("OPENAI_API_KEY")
                if key:
                    openai.api_key = key

            async def detect_language(self, text: str) -> Tuple[str, float]:
                # Use the OpenAI 'detect' pattern via embeddings or models is not
                # available as a direct single API call in all accounts. We'll fall
                # back to a simple prompt-based approach for best-effort detection.
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a language detection assistant."},
                            {"role": "user", "content": f"What language is this? Reply with 'lang: <code>' and 'confidence: <0-1>'\nText:\n{text[:1000]}"},
                        ],
                        max_tokens=20,
                    )
                    out = resp.choices[0].message.content
                    # naive parse
                    lang = "unknown"
                    conf = 0.0
                    for part in out.splitlines():
                        if part.lower().startswith("lang:"):
                            lang = part.split(":", 1)[1].strip()
                        if part.lower().startswith("confidence:"):
                            try:
                                conf = float(part.split(":", 1)[1].strip())
                            except Exception:
                                conf = 0.0
                    return lang, conf
                except Exception:
                    return await LangDetectProvider().detect_language(text)

            async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful translator. Translate the user text to the target language exactly and nothing else."},
                            {"role": "user", "content": f"Translate to {target_language}:\n\n{text[:3000]}"},
                        ],
                        max_tokens=4000,
                    )
                    translated = resp.choices[0].message.content.strip()
                    # We don't get a confidence score from OpenAI; mark as 0.9
                    return translated, 0.9
                except Exception:
                    return text, 0.0

    except Exception:
        OpenAIProvider = None


    async def translate_content(
        data_chunks: List[DocumentChunk],
        target_language: str = "en",
        translation_provider: str = "noop",
        confidence_threshold: float = 0.8,
    ) -> List[DocumentChunk]:
        """Translate non-English content to target language and attach metadata.

        For each chunk in `data_chunks` this function will:
        - detect language and store LanguageMetadata in chunk.metadata['language']
        - if detected language != target_language and confidence >= threshold,
          perform translation via the chosen provider and store TranslatedContent
          in chunk.metadata['translation'] as well as keeping original text.

        The function is defensive: if optional libraries or API keys are missing
        it will fall back to a noop provider.
        """
        # pick provider
        provider: Optional[TranslationProvider]
        if translation_provider == "openai" and OpenAIProvider is not None:
            provider = OpenAIProvider()
        elif translation_provider == "langdetect":
            provider = LangDetectProvider()
        else:
            provider = NoOpProvider()

        enhanced: List[DocumentChunk] = []

        for chunk in data_chunks:
            text = getattr(chunk, "text", "") or ""
            content_id = getattr(chunk, "id", getattr(chunk, "chunk_index", ""))
            # detect language
            try:
                lang, conf = await provider.detect_language(text)
            except Exception as e:
                logger.exception("language detection failed: %s", e)
                lang, conf = "unknown", 0.0

            requires_translation = (lang != target_language) and (conf >= confidence_threshold)
            lang_meta = LanguageMetadata(
                content_id=str(content_id),
                detected_language=lang,
                language_confidence=conf,
                requires_translation=requires_translation,
                character_count=len(text),
            )

            # attach language metadata to chunk.metadata
            chunk.metadata = getattr(chunk, "metadata", {}) or {}
            chunk.metadata["language"] = lang_meta

            # perform translation when necessary
            if requires_translation:
                try:
                    translated_text, t_conf = await provider.translate(text, target_language)
                except Exception as e:
                    logger.exception("translation failed: %s", e)
                    translated_text, t_conf = text, 0.0

                trans = TranslatedContent(
                    original_chunk_id=str(content_id),
                    original_text=text,
                    translated_text=translated_text,
                    source_language=lang,
                    target_language=target_language,
                    translation_provider=translation_provider,
                    confidence_score=t_conf,
                )
                chunk.metadata["translation"] = trans

                # Optionally, keep a translated_text attribute for quick searching
                setattr(chunk, "translated_text", translated_text)

            enhanced.append(chunk)

        return enhanced