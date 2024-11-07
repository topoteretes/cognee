from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document

class TextSummary(DataPoint):
    text: str
    chunk: DocumentChunk

    _metadata: dict = {
        "index_fields": ["text"],
    }

