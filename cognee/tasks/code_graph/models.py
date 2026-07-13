from typing import Any, Optional

from pydantic import Field

from cognee.infrastructure.engine.models.DataPoint import DataPoint


class CodeRepository(DataPoint):
    """The repository an enola snapshot was extracted from."""

    name: str
    path: str
    metadata: dict = {"index_fields": ["name"]}


class CodeGraphEntity(DataPoint):
    """Common shape of every enola fact mapped into the graph."""

    name: str
    kind: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    repo: Optional[str] = None
    description: Optional[str] = None
    fact_properties: dict[str, Any] = Field(default_factory=dict)
    part_of: Optional[CodeRepository] = None
    metadata: dict = {"index_fields": ["name"]}


class CodeModule(CodeGraphEntity):
    """A module/package (enola fact kind: module)."""


class CodeSymbol(CodeGraphEntity):
    """A code symbol (enola fact kind: symbol).

    symbol_kind is one of: function, method, struct, interface, type, class,
    variable, constant, enum.
    """

    symbol_kind: Optional[str] = None


class ApiEndpoint(CodeGraphEntity):
    """An API route (enola fact kind: route)."""


class StorageResource(CodeGraphEntity):
    """A storage resource such as a table or bucket (enola fact kind: storage)."""


class ExternalDependency(CodeGraphEntity):
    """An external dependency (enola fact kind: dependency)."""


class CodeService(CodeGraphEntity):
    """A deployable service (enola fact kind: service)."""


class CodeTestReference(CodeGraphEntity):
    """A test-to-symbol reference (enola fact kind: test_ref)."""


class CodeFileReference(CodeGraphEntity):
    """A file-level reference (enola fact kind: file_ref)."""
