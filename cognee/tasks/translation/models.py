from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk


class TranslatedContent(DataPoint):
    """
    Represents translated content with quality metrics.

    This class stores both the original and translated versions of content,
    along with metadata about the translation process including source and
    target languages, translation provider used, and confidence scores.

    Instance variables include:

    - original_chunk_id: UUID of the original document chunk
    - original_text: The original text before translation
    - translated_text: The translated text content
    - source_language: Detected or specified source language code (e.g., "es", "fr", "de")
    - target_language: Target language code for translation (default: "en")
    - translation_provider: Name of the translation service used
    - confidence_score: Translation quality/confidence score (0.0 to 1.0)
    - translation_timestamp: When the translation was performed
    - translated_from: Reference to the original DocumentChunk
    """

    original_chunk_id: UUID
    original_text: str
    translated_text: str
    source_language: str
    target_language: str = "en"
    translation_provider: str
    confidence_score: float
    translation_timestamp: datetime = None
    translated_from: Optional[DocumentChunk] = None

    metadata: dict = {"index_fields": ["source_language", "translated_text"]}

    def __init__(self, **data):
        if data.get("translation_timestamp") is None:
            data["translation_timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


class LanguageMetadata(DataPoint):
    """
    Language information for content.

    This class stores metadata about the detected language of content,
    including confidence scores and whether translation is required.

    Instance variables include:

    - content_id: UUID of the associated content
    - detected_language: ISO 639-1 language code (e.g., "en", "es", "fr")
    - language_confidence: Confidence score for language detection (0.0 to 1.0)
    - requires_translation: Whether the content needs translation
    - character_count: Number of characters in the content
    - language_name: Human-readable language name (e.g., "English", "Spanish")
    """

    content_id: UUID
    detected_language: str
    language_confidence: float
    requires_translation: bool
    character_count: int
    language_name: Optional[str] = None

    metadata: dict = {"index_fields": ["detected_language"]}
