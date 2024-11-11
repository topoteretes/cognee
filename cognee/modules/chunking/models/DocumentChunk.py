from typing import Optional
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document

class DocumentChunk(DataPoint):
    text: str
    word_count: int
    chunk_index: int
    cut_type: str
    is_part_of: Document

    _metadata: Optional[dict] = {
        "index_fields": ["text"],
    }
