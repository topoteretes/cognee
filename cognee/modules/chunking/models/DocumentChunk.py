from typing import List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity


class DocumentChunk(DataPoint):
    __tablename__ = "document_chunk"
    text: str
    word_count: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    pydantic_type: str = "DocumentChunk"
    contains: List[Entity] = None

    _metadata: dict = {"index_fields": ["text"], "type": "DocumentChunk"}
