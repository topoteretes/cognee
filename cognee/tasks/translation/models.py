from datetime import datetime
from pydantic import BaseModel

class TranslatedContent(BaseModel):
    """Represents translated content with quality metrics"""
    original_chunk_id: str
    original_text: str
    translated_text: str
    source_language: str
    target_language: str = "en"
    translation_provider: str
    confidence_score: float
    translation_timestamp: datetime
    metadata: dict = {"index_fields": ["source_language", "original_chunk_id"]}

class LanguageMetadata(BaseModel):
    """Language information for content"""
    content_id: str
    detected_language: str
    language_confidence: float
    requires_translation: bool
    character_count: int
    metadata: dict = {"index_fields": ["detected_language", "content_id"]}
