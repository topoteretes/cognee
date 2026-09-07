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
    # Projection of the snapshot's receipt.json (format_version, enola version,
    # git provenance, counts and the extraction-quality block) stamped with
    # the snapshot id, so SearchType.CODE's delta operation can report how the
    # graph was produced and how complete the extraction was.
    last_receipt: Optional[dict] = None
    metadata: dict = {"index_fields": ["name"]}


class CodeGraphEntity(DataPoint):
    """Common shape of every enola fact mapped into the graph.

    fact_hash fingerprints the derived fields, so re-ingestion can write only
    the facts whose content actually changed (delta writes).

    enola_id is the writer's own fact identity (32 hex chars over repo, kind,
    name and file; enola >= 0.4.10). Cognee's node identity stays (repo, kind,
    name) — see fact_node_id — so several enola ids can land on one node; the
    first occurrence's id is kept. Historical snapshots carry none.
    """

    name: str
    kind: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    end_line: Optional[int] = None
    repo: Optional[str] = None
    enola_id: Optional[str] = None
    description: Optional[str] = None
    fact_properties: dict[str, Any] = Field(default_factory=dict)
    fact_hash: Optional[str] = None
    part_of: Optional[CodeRepository] = None
    metadata: dict = {"index_fields": ["name"]}


class CodeModule(CodeGraphEntity):
    """A module/package (enola fact kind: module)."""


class CodeSymbol(CodeGraphEntity):
    """A code symbol (enola fact kind: symbol).

    symbol_kind is one of the contract values function, method, getter, struct,
    interface, type, class, variable, constant, enum — plus descriptive values
    newer extractors add (e.g. document/section for markdown pages).
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

    enola explainers (cycles, layers, hotspots, god-class, dependency-depth,
    exported-surface, complexity-outliers, dead-methods, unused-routes, ...)
    emit these into insights.json. Each insight is linked to the facts it
    cites via ``evidences`` edges (resolved through the evidence's ``fact_id``
    when the writer supplied one). The explainer name, confidence, the
    machine-readable ``metrics`` block and the ``informational`` flag live in
    fact_properties.
    """


class CodeIntent(CodeGraphEntity):
    """Architecture declared in enola-intent.yaml, not measured from source (kind: intent).

    intent_kind (service, seam/consumes, layer, claim, page, ...) and the
    declaration file live in fact_properties.
    """


class CodeExtractionAccount(CodeGraphEntity):
    """An extractor's own coverage account for one repo (kind: extraction).

    Named ``<extractor>:<account>`` (e.g. ``ruby:calls``); carries the
    extractor, language, edge_coverage and unresolved_* counters, so a thin
    graph can be told apart from a thin extraction.
    """


class CodeAssociation(CodeGraphEntity):
    """A framework model relationship such as Rails has_many/belongs_to (kind: association).

    Named ``Model#macro`` (or ``Child<Parent`` for an STI chain); model, macro,
    target and through live in fact_properties.
    """


class CodeLintFinding(CodeGraphEntity):
    """A finding an external linter reported through enola's provider seam (kind: lint)."""
