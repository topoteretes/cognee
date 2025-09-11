from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
from pydantic import Field, field_validator

from cognee.infrastructure.engine.models import DataPoint


class TranslatedContent(DataPoint):
    """Represents translated content with quality metrics.

    Stores the original and translated text, provider used, a confidence
    score and a timestamp. Intended to be stored as metadata on the
    originating DocumentChunk so the original and translation live
    together.
    """
    original_chunk_id: str
    original_text: str
    translated_text: str
    source_language: str
    target_language: str = "en"
    translation_provider: str = "noop"
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    translation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = Field(default_factory=lambda: {"index_fields": ["source_language", "original_chunk_id"]})


class LanguageMetadata(DataPoint):
    """Language information for content.

    Records the detected language, detection confidence, whether the
    chunk requires translation and a simple character count.
    """
    content_id: str
    detected_language: str
    language_confidence: float = Field(0.0, ge=0.0, le=1.0)
    requires_translation: bool = False
    character_count: int = Field(0, ge=0)
    metadata: Dict = Field(default_factory=lambda: {"index_fields": ["detected_language", "content_id"]})