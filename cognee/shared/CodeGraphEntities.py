from typing import List, Optional
from cognee.infrastructure.engine import DataPoint

class Repository(DataPoint):
    path: str
    type: Optional[str] = "Repository"

class CodeFile(DataPoint):
    extracted_id: str  # actually file path
    type: Optional[str] = "CodeFile"
    source_code: Optional[str] = None
    part_of: Optional[Repository] = None
    depends_on: Optional[List["CodeFile"]] = None
    depends_directly_on: Optional[List["CodeFile"]] = None
    contains: Optional[List["CodePart"]] = None

    _metadata: dict = {
        "index_fields": ["source_code"]
    }

class CodePart(DataPoint):
    type: str
    # part_of: Optional[CodeFile]
    source_code: str
    type: Optional[str] = "CodePart"

    _metadata: dict = {
        "index_fields": ["source_code"]
    }

class CodeRelationship(DataPoint):
    source_id: str
    target_id: str
    type: str  # between files
    relation: str  # depends on or depends directly

CodeFile.model_rebuild()
CodePart.model_rebuild()
