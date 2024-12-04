from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.shared.CodeGraphEntities import CodeFile


class TextSummary(DataPoint):
    __tablename__ = "text_summary"
    text: str
    made_from: DocumentChunk

    _metadata: dict = {
        "index_fields": ["text"],
    }


class CodeSummary(DataPoint):
    text: str
    made_from: CodeFile

    _metadata: dict = {
        "index_fields": ["text"],
    }
