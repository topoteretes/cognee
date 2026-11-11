"""Shared utilities for DLT ingestion."""

import json
import os
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

# A dlt source sets this attribute to opt into the "document" ingestion path:
# each row becomes a text document that flows through normal cognify (LLM entity
# extraction), instead of the default relational schema-context path. The value
# is the ``system_metadata["source"]`` tag used for that source's rows. This
# lets resolve_dlt_sources stay connector-agnostic — a connector declares its
# own nature rather than the shared engine hard-coding connector names.
DOCUMENT_SOURCE_ATTR = "cognee_document_source"

# Opt-in namespace for connectors whose incremental state must survive
# alternating destination datasets. Unrelated DLT sources keep their contract.
PIPELINE_SCOPE_ATTR = "cognee_pipeline_scope"

# The database a connection-string source reads (a SQLite file's name, a server
# database's name), so its table nodes can say which database they belong to.
SOURCE_LABEL_ATTR = "cognee_source_label"


def pipeline_name_for_source(source, dataset_name: str) -> str:
    from hashlib import sha256

    scope = getattr(source, PIPELINE_SCOPE_ATTR, None)
    if not isinstance(scope, str) or not scope:
        return "ingest_dlt_source"
    digest = sha256(json.dumps([dataset_name, scope]).encode()).hexdigest()[:32]
    return f"ingest_dlt_{digest}"


# Community/cloud hosts can refuse unsafe older cores before ingestion starts.
# Version 1 scopes cleanup by staging table and handles a confirmed empty table.
# Version 2 reads the per-row node_set column (NODE_SET_COLUMN).
# Version 3 reads the per-row structure column (STRUCTURE_COLUMN).
DOCUMENT_SYNC_VERSION = 3

# A document-mode row may carry its own node sets in this column, as a JSON
# list of names (see resolve_dlt_sources._row_node_set). Every name is
# namespaced under the source tag, so a row can never name one of cognee's
# own node sets. The type hint is applied at load time so dlt stores the list
# on the row as json instead of normalizing it into a child table.
NODE_SET_COLUMN = "cognee_node_set"
NODE_SET_COLUMN_HINT = {NODE_SET_COLUMN: {"data_type": "json", "nullable": True}}

# A document-mode row may say where it sits in its source's tree in this column:
# ``{"ancestors": [{"kind", "id", "name"?, "document"?}, ...]}``, nearest parent
# first, empty for a row at the top. A ``"document": true`` entry is another row of
# this source and table (its ``id`` column) and ends the list; any other entry is a
# container with no row of its own (a database, a folder). Never hashed, so a move
# keeps the row's data_id. Loaded as json for the same reason as the node set column.
STRUCTURE_COLUMN = "cognee_structure"
STRUCTURE_COLUMN_HINT = {STRUCTURE_COLUMN: {"data_type": "json", "nullable": True}}
DOCUMENT_COLUMN_HINTS = {**NODE_SET_COLUMN_HINT, **STRUCTURE_COLUMN_HINT}


def guarded_rows(rows, check_active=None):
    """Check authorization before each extraction step and before publishing it.

    Hosts run extraction on a worker thread and supply a synchronous bridge to
    their credential store. Standalone SDK sources need no such callback.
    """
    iterator = iter(rows)
    while True:
        if check_active is not None:
            check_active()
        try:
            row = next(iterator)
        except StopIteration:
            if check_active is not None:
                check_active()
            return
        if check_active is not None:
            check_active()
        yield row


def with_retries(
    attempt: Callable[[], _T],
    *,
    retry_on: tuple[type[BaseException], ...],
    delay: Callable[[BaseException, int], float | None],
    attempts: int,
    check_active: Callable[[], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> _T:
    """Run one provider request, retried by the connector's own policy.

    ``attempt`` makes one request. An error of a ``retry_on`` class is retried after
    ``delay(error, number)`` seconds unless it was the last of ``attempts`` or ``delay`` answers
    ``None``; then it is raised as it is. ``check_active`` runs before every wait and every
    retry, never before a first attempt, and what it raises propagates as it is.
    """
    for number in range(attempts):
        if number and check_active is not None:
            check_active()
        try:
            return attempt()
        except retry_on as error:
            if number == attempts - 1:
                raise
            wait = delay(error, number)
            if wait is None:
                raise
        if check_active is not None:
            check_active()
        (sleep or time.sleep)(wait)
    raise ValueError("with_retries needs at least one attempt")


def document_source_tag(item) -> str | None:
    """Return the document-source tag a dlt source opted into, else ``None``."""
    tag = getattr(item, DOCUMENT_SOURCE_ATTR, None)
    return tag if isinstance(tag, str) and tag else None


def document_write_disposition(source) -> str | None:
    """The write disposition a document source runs under when the caller names none.

    ``None``, each resource's own, when every selected resource declares
    ``merge``; ``"replace"`` otherwise. dlt reports a resource that declares
    nothing as ``append``, the same as one that declares it, so only a declared
    ``merge`` can be told apart and kept; ``append`` needs the caller's argument.
    """
    selected = getattr(source, "selected_resources", None)
    resources = list(selected.values()) if selected is not None else [source]
    return None if resources and all(_declares_merge(r) for r in resources) else "replace"


def _declares_merge(resource) -> bool:
    disposition = getattr(resource, "write_disposition", None)
    if isinstance(disposition, dict):  # merge with a strategy: {"disposition": "merge", ...}
        disposition = disposition.get("disposition")
    return disposition == "merge"


def apply_dlt_runtime_defaults() -> None:
    """Set cognee's defaults for dlt's own settings, as environment variables.

    Runs when ``ingest_dlt_source`` is imported, before anything reads dlt's
    configuration: dlt starts its telemetry on the first read (building a
    ``sql_database`` source is one) and never re-reads the setting. A value already in the
    environment is kept; one set only in dlt's ``config.toml`` is not seen
    (reading it would start the telemetry) and is overridden. A process that
    used dlt before importing cognee has already started dlt's telemetry, and
    this does not stop it: import cognee first, or set the variable yourself.

    - ``RUNTIME__DLTHUB_TELEMETRY=false`` when ``TELEMETRY_DISABLED`` is set, so
      cognee's switch also stops the events dlt sends to its own endpoint.
    - ``LOAD__DELETE_COMPLETED_JOBS=true``: completed load packages hold every
      row's full text under dlt's pipelines directory, which no forget reaches.
    """
    if os.getenv("TELEMETRY_DISABLED"):
        os.environ.setdefault("RUNTIME__DLTHUB_TELEMETRY", "false")
    os.environ.setdefault("LOAD__DELETE_COMPLETED_JOBS", "true")


def metadata_source(metadata) -> str | None:
    """Extract the ``source`` field from system metadata.

    Accepts a dict, a JSON string, or an object with a ``system_metadata``
    attribute (a Data record / DataItem). Returns None when the source cannot
    be determined. Deliberately never reads external_metadata: that field is
    user-writable, and routing/deletion decisions must not key on user bytes.

    Shared reader for every system_metadata["source"] check (DLT here, code
    files in cognee.tasks.code_graph.code_files) so the tag is parsed one way.
    """
    meta = getattr(metadata, "system_metadata", metadata)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(meta, dict):
        return meta.get("source")
    return None


def is_dlt_sourced(metadata) -> bool:
    """Check whether system_metadata indicates a legacy per-row DLT item (source == "dlt")."""
    return metadata_source(metadata) == "dlt"


def is_dlt_source_manifest(metadata) -> bool:
    """Check whether system_metadata indicates a DLT source manifest (source == "dlt_source")."""
    return metadata_source(metadata) == "dlt_source"


async def load_dlt_manifest(raw_data_location: str) -> dict:
    """Load a DLT source manifest from storage.

    Single reader of the DLT source manifest format written by
    ``resolve_dlt_sources._build_source_manifest_item``.
    """
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(raw_data_location, mode="r", encoding="utf-8") as file:
        return json.loads(file.read())


def column_selected(selection: dict | None, table_name: str, column: str) -> bool:
    """Whether ``selection`` ({table: [column, ...]}, "*" wildcards on either side)
    names this cell. A table named in the selection gets exactly its list — an
    empty list means none for that table — and only an unnamed table takes the
    wildcard. An empty or missing selection names nothing."""
    if not selection:
        return False
    columns = selection[table_name] if table_name in selection else selection.get("*", [])
    return "*" in columns or column in columns
