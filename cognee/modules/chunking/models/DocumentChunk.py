from typing import List, Optional

from pydantic import Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity


class DocumentChunk(DataPoint):
    __tablename__ = "document_chunk"
    text: str
    word_count: int
    token_count: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    pydantic_type: str = "DocumentChunk"
    contains: List[Entity] = None
    # potential_nodes: List[str] = Field(default_factory=list)
    # potential_relationships: List[str] = Field(default_factory=list)

    _metadata: dict = {"index_fields": ["text"], "type": "DocumentChunk"}
