from typing import Optional
from pydantic import Field
from cognee.low_level import DataPoint


class CodeFile(DataPoint):
    """A source file ingested into the code graph."""

    file_path: str = Field(..., description="Repo-relative or absolute path of the source file")
    language: str = "python"
    metadata: dict = {"index_fields": ["file_path"], "identity_fields": ["file_path"]}


class CodeFunction(DataPoint):
    """A function or method definition."""

    name: str = Field(..., description="Fully-qualified function name")
    file_path: str = Field(..., description="Source file containing this function")
    start_line: int = 0
    end_line: int = 0
    docstring: Optional[str] = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name", "file_path"]}


class CodeClass(DataPoint):
    """A class definition."""

    name: str = Field(..., description="Fully-qualified class name")
    file_path: str = Field(..., description="Source file containing this class")
    start_line: int = 0
    end_line: int = 0
    docstring: Optional[str] = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name", "file_path"]}


class CodeImport(DataPoint):
    """An import relationship between a file and a module."""

    importer_path: str = Field(..., description="Path of the file that imports")
    imported_module: str = Field(..., description="Module being imported")
    metadata: dict = {
        "index_fields": ["imported_module"],
        "identity_fields": ["importer_path", "imported_module"],
    }
