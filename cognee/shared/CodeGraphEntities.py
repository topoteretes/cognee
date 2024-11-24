from typing import Any, List, Literal, Optional, Union

from cognee.infrastructure.engine import DataPoint


class Repository(DataPoint):
    path: str


class CodeFile(DataPoint):
    extracted_id: str  # actually file path
    type: str
    source_code: str

    _metadata: dict = {
        "index_fields": ["source_code"]
    }

class CodeRelationship(DataPoint):
    source_id: str
    target_id: str
    type: str  # between files
    relation: str  # depends on or depends directly
