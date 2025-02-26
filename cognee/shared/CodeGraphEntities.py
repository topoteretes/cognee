from typing import List, Optional
from cognee.low_level import DataPoint


class Repository(DataPoint):
    path: str


class ImportStatement(DataPoint):
    name: str
    module: str
    start_point: tuple
    end_point: tuple
    source_code: str
    file_path: Optional[str] = None


class FunctionDefinition(DataPoint):
    name: str
    start_point: tuple
    end_point: tuple
    source_code: str
    file_path: Optional[str] = None
    metadata: dict = {"index_fields": ["source_code"]}


class ClassDefinition(DataPoint):
    name: str
    start_point: tuple
    end_point: tuple
    source_code: str
    file_path: Optional[str] = None
    metadata: dict = {"index_fields": ["source_code"]}


class CodeFile(DataPoint):
    name: str
    file_path: str
    source_code: Optional[str] = None
    part_of: Optional[Repository] = None
    depends_on: Optional[List["ImportStatement"]] = []
    provides_function_definition: Optional[List["FunctionDefinition"]] = []
    provides_class_definition: Optional[List["ClassDefinition"]] = []
    metadata: dict = {"index_fields": ["name"]}


class CodePart(DataPoint):
    file_path: str
    source_code: Optional[str] = None
    metadata: dict = {"index_fields": []}


class SourceCodeChunk(DataPoint):
    code_chunk_of: Optional[CodePart] = None
    source_code: Optional[str] = None
    previous_chunk: Optional["SourceCodeChunk"] = None
    metadata: dict = {"index_fields": ["source_code"]}


CodeFile.model_rebuild()
SourceCodeChunk.model_rebuild()
