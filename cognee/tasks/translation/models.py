from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field

class TranslatedContent(BaseModel):
    original_chunk_id: str
    original_text: str
    translated_text: str
    source_language: str
    target_language: str = "en"
    translation_provider: str
    confidence_score: float
    translation_timestamp: datetime
    metadata: Dict[str, List[str]] = Field(
        default_factory=lambda: {"index_fields": ["source_language", "original_chunk_id"]}
    )

class LanguageMetadata(BaseModel):
    content_id: str
    detected_language: str
    language_confidence: float
    requires_translation: bool
    character_count: int
    metadata: Dict[str, List[str]] = Field(
        default_factory=lambda: {"index_fields": ["detected_language", "content_id"]}
    )
