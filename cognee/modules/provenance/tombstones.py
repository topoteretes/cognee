"""Deletion-driven tombstones for the audit ledger, one per data item.

The ledger is append-only and holds one row per data item: every writer
records a ``data_item`` row for the (dataset, data item) it stores output for
(``record_provenance``). When that item is deleted this module invalidates
that row (``prov:Invalidation``). The nodes and edges the item produced are
not in the ledger; the graph's own source refs say which data item each came
from, so a tombstone needs only ``(dataset_id, data_id)``.

A ``data_deleted`` / ``dataset_deleted`` tombstone says the document is gone,
so it is written last: after the ``Data`` row was removed and, for a single
item, checked to be gone. A delete that fails earlier leaves the row live.
Whatever removes only an item's memory keeps the document and writes no
tombstone: ``forget(memory_only=True)``, a DLT source purged for re-emission,
and the rollback of a failed pipeline run. The row stays live for as long as
the document is stored, so the delete that finally removes it is the one the
ledger records.

Two scopes:

- **data item**: ``forget(data_id=...)`` / ``datasets.delete_data``, and the
  DLT orphan sweep for the rows it removed.
- **dataset**: every live data item of a deleted dataset.

Only rows that exist and are live are invalidated (the ledger is opt-in, so
most deletes find nothing). A tombstone that cannot be written raises, so the
delete fails with it. Deleting the same data item again writes it: a delete
that finds no ``Data`` row still retracts a live ledger row for the id. A
deleted dataset cannot be deleted again, so after such a failure its rows
stay live.
"""

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from . import storage
from .manager import get_provenance_manager

logger = get_logger("provenance.tombstones")

# Tombstones per chained transaction. Bounds transaction size on dataset
# deletes; a chunk of 1000 is the same order as the writer's per-batch size.
_COMMIT_CHUNK = 1000


def ledger_node_key(dataset_id: UUID | str, node_id: str) -> str:
    """Ledger key of a graph node, as ``record_provenance`` namespaces it."""
    return f"{dataset_id}:{node_id}"


def ledger_edge_key(
    dataset_id: UUID | str, source_id: str, target_id: str, relationship_name: str
) -> str:
    """Ledger key of a graph edge, as ``record_provenance`` namespaces it."""
    return (
        f"rel:{ledger_node_key(dataset_id, source_id)}:{relationship_name}:"
        f"{ledger_node_key(dataset_id, target_id)}"
    )


def ledger_data_item_key(dataset_id: UUID | str, data_id: UUID | str) -> str:
    """Ledger key of a data item, as ``record_provenance`` writes it."""
    return f"{dataset_id}:data_item:{data_id}"


def dataset_id_from_ledger_key(entity_id: str) -> UUID | None:
    """The dataset that owns a ledger key, or None for an unscoped key.

    Inverse of the ``ledger_*_key`` builders — the read surface
    uses it to decide whose ACL governs a lookup. Archive suffixes do not
    matter: the dataset id is always the first (or, for edges, second)
    colon-separated segment.
    """
    if not entity_id:
        return None
    parts = entity_id.split(":", 2)
    candidate = parts[1] if parts[0] == "rel" and len(parts) > 1 else parts[0]
    try:
        return UUID(candidate)
    except (ValueError, AttributeError, TypeError):
        return None


def agent_id_for(user: Any) -> str:
    """Who a ledger row is attributed to: the user's id, else ``"cognee"``.

    The id, never the email: the ledger is kept after the data it describes is
    deleted, so it holds ids and hashes only.
    """
    user_id = getattr(user, "id", None)
    return str(user_id) if user_id is not None else "cognee"


async def _tombstone_ids(
    entity_ids: Iterable[str],
    *,
    agent_id: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    manager = get_provenance_manager()
    written = 0
    batch = manager.batch()
    for entity_id in entity_ids:
        batch.invalidate(entity_id, agent_id=agent_id, reason=reason, metadata=metadata)
        if len(batch) >= _COMMIT_CHUNK:
            written += len(await batch.commit())
    if len(batch):
        written += len(await batch.commit())
    return written


async def tombstone_data_item(
    dataset_id: UUID,
    data_id: UUID,
    *,
    user: Any = None,
    reason: str = "data_deleted",
) -> int:
    """Tombstone the ledger row of a deleted data item.

    Returns the number of tombstones written: 1, or 0 when the ledger holds no
    live row for the item (never recorded, or already tombstoned).
    """
    return await tombstone_data_items(dataset_id, [data_id], user=user, reason=reason)


async def tombstone_data_items(
    dataset_id: UUID,
    data_ids: Iterable[UUID],
    *,
    user: Any = None,
    reason: str = "data_deleted",
) -> int:
    """Tombstone the ledger rows of deleted data items of one dataset.

    One lookup and one chained write for the whole list, for a sweep that
    removes many rows at once. Returns the number of tombstones written.
    """
    live_ids = await storage.retrieve_live_ids(
        [ledger_data_item_key(dataset_id, data_id) for data_id in data_ids]
    )
    if not live_ids:
        return 0
    written = await _tombstone_ids(live_ids, agent_id=agent_id_for(user), reason=reason)
    logger.info("Tombstoned %d data item(s) of dataset %s in the ledger", written, dataset_id)
    return written


async def tombstone_dataset(
    dataset_id: UUID,
    *,
    user: Any = None,
    reason: str = "dataset_deleted",
) -> int:
    """Tombstone every live data item of the dataset. Returns the number written."""
    # Collected before writing: the prefix scan must not observe its own
    # in-flight invalidations.
    pending = [
        entity_id
        async for entity_id in storage.iter_live_with_prefix(ledger_data_item_key(dataset_id, ""))
    ]
    if not pending:
        return 0
    written = await _tombstone_ids(pending, agent_id=agent_id_for(user), reason=reason)
    logger.info("Tombstoned %d data item(s) in the ledger for dataset %s", written, dataset_id)
    return written
