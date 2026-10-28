"""Shared utilities for DLT ingestion.

Moved from cognee.tasks.ingestion.dlt_utils (cognee dev, 2026-10-26). This is
the one definition both cognee's ingestion and the connector packages read: the
string values are byte-identical to cognee's and must stay so, because cognee
reads them by attribute and column name. ``load_dlt_manifest`` stayed in cognee
— it reads cognee's storage, which is a host concern.

One value is not a name: ``DOCUMENT_SYNC_VERSION`` is the document-sync protocol
version, which moves with cognee releases. The copy here is the version this
package was cut at; a source's capability gate ("does the host read per-row
node sets?") must read the *host's* value through
:func:`host_document_sync_version`, never this constant, or the gate compares
the package with itself and never refuses an old cognee.
"""

import importlib
import json

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
#
# In cognee this constant IS the host's capability. Here it is the version the
# sources in this repo were written against; see host_document_sync_version()
# for the value a gate must compare with.
DOCUMENT_SYNC_VERSION = 3


def host_document_sync_version() -> int:
    """The document-sync protocol version the host's cognee implements.

    Reads ``cognee.tasks.ingestion.dlt_utils.DOCUMENT_SYNC_VERSION`` from the
    installed cognee; ``0`` for a cognee too old to have DLT document
    ingestion at all. Without cognee it returns this package's own
    :data:`DOCUMENT_SYNC_VERSION`: there is no ingestion to be too old, and a
    connector's own tests run without one. Imported lazily, so this module
    stays importable with cognee absent.

    This — not the constant above — is what a source's capability gate reads.
    Cognee's sources did ``getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0)``
    against cognee's own module; ported to this package that would compare
    the package with itself and pass on every host.
    """
    try:
        importlib.import_module("cognee")
    except ImportError:
        return DOCUMENT_SYNC_VERSION
    try:
        host = importlib.import_module("cognee.tasks.ingestion.dlt_utils")
    except ImportError:
        return 0
    return int(getattr(host, "DOCUMENT_SYNC_VERSION", 0) or 0)


def require_document_sync_version(minimum: int, *, provider: str, feature: str) -> None:
    """Refuse a sync the host's cognee cannot ingest correctly.

    The gate every source in cognee carried, shared: ``provider`` and
    ``feature`` fill the message cognee's sources used (``"Notion sync requires
    a Cognee build that reads per-row node sets and structure
    (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."``). Call it
    where the source factory starts, before any provider call.
    """
    if host_document_sync_version() < minimum:
        raise RuntimeError(
            f"{provider} sync requires a Cognee build that {feature} "
            f"(DOCUMENT_SYNC_VERSION >= {minimum}). Upgrade Cognee before syncing."
        )


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


def document_source_tag(item) -> str | None:
    """Return the document-source tag a dlt source opted into, else ``None``."""
    tag = getattr(item, DOCUMENT_SOURCE_ATTR, None)
    return tag if isinstance(tag, str) and tag else None


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


def column_selected(selection: dict | None, table_name: str, column: str) -> bool:
    """Whether ``selection`` ({table: [column, ...]}, "*" wildcards on either side)
    names this cell. A table named in the selection gets exactly its list — an
    empty list means none for that table — and only an unnamed table takes the
    wildcard. An empty or missing selection names nothing."""
    if not selection:
        return False
    columns = selection[table_name] if table_name in selection else selection.get("*", [])
    return "*" in columns or column in columns


__all__ = [
    "DOCUMENT_COLUMN_HINTS",
    "DOCUMENT_SOURCE_ATTR",
    "DOCUMENT_SYNC_VERSION",
    "NODE_SET_COLUMN",
    "NODE_SET_COLUMN_HINT",
    "PIPELINE_SCOPE_ATTR",
    "STRUCTURE_COLUMN",
    "STRUCTURE_COLUMN_HINT",
    "column_selected",
    "document_source_tag",
    "guarded_rows",
    "host_document_sync_version",
    "is_dlt_source_manifest",
    "is_dlt_sourced",
    "metadata_source",
    "pipeline_name_for_source",
    "require_document_sync_version",
]
