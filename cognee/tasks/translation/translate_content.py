from __future__ import annotations

from typing import List, Protocol, Tuple, Optional
import logging
import asyncio
import os

from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata
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
            return "en", 0.5  # Lower confidence to avoid false positives
        except UnicodeEncodeError:
            return "unknown", 0.4

    async def translate(self, text: str, _target_language: str) -> Tuple[str, float]:
        # No translation performed
        return text, 0.0


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
    import openai

    class OpenAIProvider:
        def __init__(self, model: Optional[str] = None, timeout: float = 30.0):
            # Prefer modern client; fall back to legacy globals.
            self.model = model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
            self.timeout = float(os.getenv("OPENAI_TIMEOUT", timeout))
            key = os.getenv("OPENAI_API_KEY")
            # If the new client exists, use it; otherwise fall back to global api_key
            self._client = getattr(openai, "OpenAI", None)
            if self._client:
                # instantiate client with key if provided
                self._client = self._client(api_key=key) if key else self._client()
            elif key and hasattr(openai, "api_key"):
                openai.api_key = key

        async def detect_language(self, text: str) -> Tuple[str, float]:
            try:
                if self._client:
                    resp = await asyncio.to_thread(
                        self._client.chat.completions.create,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a language detection assistant."},
                            {"role": "user", "content": f"What language is this? Reply with 'lang: <code>' and 'confidence: <0-1>'\nText:\n{text[:1000]}"},
                        ],
                        timeout=self.timeout,
                    )
                else:
                    resp = await asyncio.to_thread(
                        openai.ChatCompletion.create,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a language detection assistant."},
                            {"role": "user", "content": f"What language is this? Reply with 'lang: <code>' and 'confidence: <0-1>'\nText:\n{text[:1000]}"},
                        ],
                        max_tokens=20,
                        request_timeout=self.timeout,
                    )
                out = resp.choices[0].message.content or ""
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
                if self._client:
                    resp = await asyncio.to_thread(
                        self._client.chat.completions.create,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful translator. Translate the user text to the target language exactly and nothing else."},
                            {"role": "user", "content": f"Translate to {target_language}:\n\n{text}"},
                        ],
                        timeout=self.timeout,
                    )
                else:
                    resp = await asyncio.to_thread(
                        openai.ChatCompletion.create,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful translator. Translate the user text to the target language exactly and nothing else."},
                            {"role": "user", "content": f"Translate to {target_language}:\n\n{text}"},
                        ],
                        max_tokens=2000,
                        request_timeout=self.timeout,
                    )
                translated = (resp.choices[0].message.content or "").strip()
                return translated, 0.9
            except Exception:
                return text, 0.0

except Exception:
    OpenAIProvider = None


def _get_provider(name: str) -> TranslationProvider:
    """Get translation provider by name."""
    name = (name or "noop").lower()
    if name == "openai" and OpenAIProvider is not None:
        return OpenAIProvider()
    if name == "langdetect":
        return LangDetectProvider()
    return NoOpProvider()


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
      in chunk.metadata['translation'] and update chunk.text with translated content.

    The function is defensive: if optional libraries or API keys are missing
    it will fall back to a noop provider.
    """
    provider = _get_provider(translation_provider)
    enhanced: List[DocumentChunk] = []

    for chunk in data_chunks:
        text = getattr(chunk, "text", "") or ""
        content_id = getattr(chunk, "id", getattr(chunk, "chunk_index", ""))
        
        # detect language
        try:
            lang, conf = await provider.detect_language(text)
        except Exception:
            logger.exception("language detection failed for content_id=%s", content_id)
            lang, conf = "unknown", 0.0
        
        # Normalize for comparisons/metrics
        lang = (lang or "unknown").lower()

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
        chunk.metadata["language"] = lang_meta.model_dump()

        # perform translation when necessary
        if requires_translation:
            try:
                translated_text, t_conf = await provider.translate(text, target_language)
            except Exception:
                logger.exception("translation failed for content_id=%s", content_id)
                translated_text, t_conf = text, 0.0

            # Determine the actual provider that handled the translation
            if OpenAIProvider is not None and isinstance(provider, OpenAIProvider):
                effective_provider = "openai"
            elif hasattr(provider, '__class__') and 'LangDetectProvider' in provider.__class__.__name__:
                effective_provider = "langdetect"
            else:
                effective_provider = "noop"

            trans = TranslatedContent(
                original_chunk_id=str(content_id),
                original_text=text,
                translated_text=translated_text,
                source_language=lang,
                target_language=target_language,
                translation_provider=effective_provider,
                confidence_score=t_conf,
            )
            if translated_text != text:
                chunk.metadata["translation"] = trans.model_dump()
                # Use translated content for subsequent tasks
                chunk.text = translated_text
            else:
                logger.info(
                    "Skipping translation metadata; provider returned unchanged text for content_id=%s",
                    content_id,
                )

        enhanced.append(chunk)

    return enhanced