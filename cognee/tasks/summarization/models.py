from typing import Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.CodeGraphEntities import CodeFile, CodePart, SourceCodeChunk


class TextSummary(DataPoint):
    text: str
    made_from: DocumentChunk

    metadata: dict = {"index_fields": ["text"]}


class CodeSummary(DataPoint):
    text: str
    summarizes: Union[CodeFile, CodePart, SourceCodeChunk]

    metadata: dict = {"index_fields": ["text"]}
