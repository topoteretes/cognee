from typing import List, Optional
from cognee.infrastructure.engine import DataPoint


class Repository(DataPoint):
    __tablename__ = "Repository"
    path: str
    _metadata: dict = {
        "index_fields": [],
        "type": "Repository"
    }

class CodeFile(DataPoint):
    __tablename__ = "codefile"
    extracted_id: str  # actually file path
    source_code: Optional[str] = None
    part_of: Optional[Repository] = None
    depends_on: Optional[List["CodeFile"]] = None
    depends_directly_on: Optional[List["CodeFile"]] = None
    contains: Optional[List["CodePart"]] = None
    _metadata: dict = {
        "index_fields": [],
        "type": "CodeFile"
    }

class CodePart(DataPoint):
    __tablename__ = "codepart"
    # part_of: Optional[CodeFile] = None
    source_code: Optional[str] = None
    _metadata: dict = {
        "index_fields": [],
        "type": "CodePart"
    }

class SourceCodeChunk(DataPoint):
    __tablename__ = "sourcecodechunk"
    code_chunk_of: Optional[CodePart] = None
    source_code: Optional[str] = None
    previous_chunk: Optional["SourceCodeChunk"] = None

    _metadata: dict = {
        "index_fields": ["source_code"],
        "type": "SourceCodeChunk"
    }

CodeFile.model_rebuild()
CodePart.model_rebuild()
SourceCodeChunk.model_rebuild()