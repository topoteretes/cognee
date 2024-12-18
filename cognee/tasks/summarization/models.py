from typing import Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.CodeGraphEntities import CodeFile, CodePart, SourceCodeChunk


class TextSummary(DataPoint):
    __tablename__ = "text_summary"
    text: str
    made_from: DocumentChunk

    _metadata: dict = {
        "index_fields": ["text"],
        "type": "TextSummary"
    }


class CodeSummary(DataPoint):
    __tablename__ = "code_summary"
    text: str
    made_from: Union[CodeFile, CodePart, SourceCodeChunk]

    _metadata: dict = {
        "index_fields": ["text"],
        "type": "CodeSummary"
    }
