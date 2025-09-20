
from __future__ import annotations
# Translation response model for structured output
from pydantic import BaseModel

class TranslationResponse(BaseModel):
    """Response model for LLM-based translation."""
    translated_text: str

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import Field

from cognee.infrastructure.engine import DataPoint


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
    source_language: str = Field(..., pattern=r"^[a-z]{2}(-[A-Z]{2})?$|^(unknown|und)$")
    target_language: str = Field("en", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    translation_provider: str = "noop"
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    translation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Inherit `metadata` from DataPoint to keep typing and defaults consistent.


class LanguageMetadata(DataPoint):
    """Language information for content.

    Records the detected language, detection confidence, whether the
    chunk requires translation and a simple character count.
    """
    content_id: str
    detected_language: str = Field(..., pattern=r"^[a-z]{2}(-[A-Z]{2})?$|^(unknown|und)$")
    language_confidence: float = Field(0.0, ge=0.0, le=1.0)
    requires_translation: bool = False
    character_count: int = Field(0, ge=0)
    # Inherit `metadata` from DataPoint to keep typing and defaults consistent.