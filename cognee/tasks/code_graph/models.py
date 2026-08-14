from typing import Any, Optional

from pydantic import Field

from cognee.infrastructure.engine.models.DataPoint import DataPoint


class CodeRepository(DataPoint):
    """The repository an enola snapshot was extracted from.

    last_snapshot_id records the enola snapshot identity of the last fully
    loaded (and swept) ingestion; extract_code_graph skips re-loading when the
    current snapshot carries the same id. It is stamped only after a load
    completes, so a crashed run can never be mistaken for an up-to-date one.
    """

    name: str
    path: str
    last_snapshot_id: Optional[str] = None
    last_delta: Optional[dict] = None
    metadata: dict = {"index_fields": ["name"]}


class CodeGraphEntity(DataPoint):
    """Common shape of every enola fact mapped into the graph.

    fact_hash fingerprints the derived fields, so re-ingestion can write only
    the facts whose content actually changed (delta writes).
    """

    name: str
    kind: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    repo: Optional[str] = None
    description: Optional[str] = None
    fact_properties: dict[str, Any] = Field(default_factory=dict)
    fact_hash: Optional[str] = None
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


class CodeInsight(CodeGraphEntity):
    """An architecture finding from an enola explainer (synthesized kind: insight).

    enola 0.3.x explainers (hotspots, god-class, dependency-depth, cycles,
    layers, exported-surface, complexity-outliers, ...) emit these into
    insights.json. Each insight is linked to the facts it cites via
    ``evidences`` edges. The explainer name and confidence live in
    fact_properties ("source", "confidence").
    """
