from typing import List, Optional

from cognee.infrastructure.engine import DataPoint


class Repository(DataPoint):
    __tablename__ = "Repository"
    path: str
    _metadata: dict = {
        "index_fields": ["source_code"],
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
        "index_fields": ["source_code"],
        "type": "CodeFile"
    }

class CodePart(DataPoint):
    __tablename__ = "codepart"
    # part_of: Optional[CodeFile]
    source_code: str
    
    _metadata: dict = {
        "index_fields": ["source_code"],
        "type": "CodePart"
    }

class CodeRelationship(DataPoint):
    source_id: str
    target_id: str
    relation: str  # depends on or depends directly
    _metadata: dict = {
        "type": "CodeRelationship"
    }

CodeFile.model_rebuild()
CodePart.model_rebuild()
