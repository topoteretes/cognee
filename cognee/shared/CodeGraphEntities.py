from typing import List, Optional
from cognee.infrastructure.engine import DataPoint


class Repository(DataPoint):
    path: str
    metadata: dict = {"index_fields": []}


class CodeFile(DataPoint):
    extracted_id: str  # actually file path
    source_code: Optional[str] = None
    part_of: Optional[Repository] = None
    depends_on: Optional[List["CodeFile"]] = None
    depends_directly_on: Optional[List["CodeFile"]] = None
    contains: Optional[List["CodePart"]] = None
    metadata: dict = {"index_fields": []}


class CodePart(DataPoint):
    file_path: str  # file path
    # part_of: Optional[CodeFile] = None
    source_code: Optional[str] = None
    metadata: dict = {"index_fields": []}


class SourceCodeChunk(DataPoint):
    code_chunk_of: Optional[CodePart] = None
    source_code: Optional[str] = None
    previous_chunk: Optional["SourceCodeChunk"] = None

    metadata: dict = {"index_fields": ["source_code"]}


CodeFile.model_rebuild()
CodePart.model_rebuild()
SourceCodeChunk.model_rebuild()
