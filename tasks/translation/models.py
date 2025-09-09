from datetime import datetime
from typing import Dict
from cognee.core.datapoint import DataPoint

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
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
    confidence_score: float = 0.0
    translation_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=lambda: {"index_fields": ["source_language", "original_chunk_id"]})

@dataclass
class LanguageMetadata(DataPoint):
    """Language information for content.

    Records the detected language, detection confidence, whether the
    chunk requires translation and a simple character count.
    """
    content_id: str
    detected_language: str
    language_confidence: float = 0.0
    requires_translation: bool = False
    character_count: int = 0
    metadata: Dict = field(default_factory=lambda: {"index_fields": ["detected_language", "content_id"]})