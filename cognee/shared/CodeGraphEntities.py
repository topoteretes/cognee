from cognee.low_level import DataPoint


class Repository(DataPoint):
    path: str


class ImportStatement(DataPoint):
    name: str
    module: str
    start_point: tuple
    end_point: tuple
    source_code: str
    file_path: str | None = None


class FunctionDefinition(DataPoint):
    name: str
    start_point: tuple
    end_point: tuple
    source_code: str
    file_path: str | None = None
    metadata: dict = {"index_fields": ["source_code"]}


class ClassDefinition(DataPoint):
    name: str
    start_point: tuple
    end_point: tuple
    source_code: str
    file_path: str | None = None
    metadata: dict = {"index_fields": ["source_code"]}


class CodeFile(DataPoint):
    name: str
    file_path: str
    language: str | None = None  # e.g., 'python', 'javascript', 'java', etc.
    source_code: str | None = None
    part_of: Repository | None = None
    depends_on: list["ImportStatement"] | None = []
    provides_function_definition: list["FunctionDefinition"] | None = []
    provides_class_definition: list["ClassDefinition"] | None = []
    metadata: dict = {"index_fields": ["name"]}


class CodePart(DataPoint):
    file_path: str
    source_code: str | None = None
    metadata: dict = {"index_fields": []}


class SourceCodeChunk(DataPoint):
    code_chunk_of: CodePart | None = None
    source_code: str | None = None
    previous_chunk: "SourceCodeChunk | None" = None
    metadata: dict = {"index_fields": ["source_code"]}


CodeFile.model_rebuild()
SourceCodeChunk.model_rebuild()
