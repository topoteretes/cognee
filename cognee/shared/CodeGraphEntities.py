from typing import List, Optional
from cognee.infrastructure.engine import DataPoint


class Repository(DataPoint):
    __tablename__ = "Repository"
    path: str
    pydantic_type: str = "Repository"
    metadata: dict = {"index_fields": []}


class CodeFile(DataPoint):
    __tablename__ = "codefile"
    extracted_id: str  # actually file path
    pydantic_type: str = "CodeFile"
    source_code: Optional[str] = None
    part_of: Optional[Repository] = None
    depends_on: Optional[List["CodeFile"]] = None
    depends_directly_on: Optional[List["CodeFile"]] = None
    contains: Optional[List["CodePart"]] = None
    metadata: dict = {"index_fields": []}


class CodePart(DataPoint):
    __tablename__ = "codepart"
    file_path: str  # file path
    # part_of: Optional[CodeFile] = None
    pydantic_type: str = "CodePart"
    source_code: Optional[str] = None
    metadata: dict = {"index_fields": []}


class SourceCodeChunk(DataPoint):
    __tablename__ = "sourcecodechunk"
    code_chunk_of: Optional[CodePart] = None
    source_code: Optional[str] = None
    pydantic_type: str = "SourceCodeChunk"
    previous_chunk: Optional["SourceCodeChunk"] = None

    metadata: dict = {"index_fields": ["source_code"]}


CodeFile.model_rebuild()
CodePart.model_rebuild()
SourceCodeChunk.model_rebuild()
