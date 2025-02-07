from typing import Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.CodeGraphEntities import CodeFile, CodePart, SourceCodeChunk


class TextSummary(DataPoint):
    __tablename__ = "text_summary"
    text: str
    made_from: DocumentChunk

    metadata: dict = {"index_fields": ["text"]}


class CodeSummary(DataPoint):
    __tablename__ = "code_summary"
    text: str
    summarizes: Union[CodeFile, CodePart, SourceCodeChunk]
    pydantic_type: str = "CodeSummary"

    metadata: dict = {"index_fields": ["text"]}
