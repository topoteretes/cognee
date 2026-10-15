"""Deletion-driven tombstones for the audit ledger.

The ledger is append-only: when ``forget()`` / ``datasets.delete_data`` /
``datasets.empty_dataset`` hard-delete graph artifacts, their ledger rows must
not keep asserting them. This module is the one place that turns a graph
deletion into ``invalidate`` tombstones (``prov:Invalidation``), and it is
called from the two delete choke points in ``cognee/modules/graph/methods``
(``delete_data_nodes_and_edges`` and ``delete_dataset_nodes_and_edges``) so
every public delete surface — including the DLT orphan purge — gets it.

Keys mirror the writer (``cognee/tasks/provenance/record_provenance.py``):
every id is namespaced by the dataset (``"{dataset_id}:{raw_id}"``) and a
relationship is ``"rel:{src}:{name}:{dst}"`` over the namespaced endpoints.

Two granularities:

- **data item**: tombstone exactly the elements the graph delete reports as
  hard-deleted (``DeletedGraphElements``). An entity that survives because
  another document of the dataset still owns it is *not* tombstoned, which is
  what the graph says too. Nothing is matched by ``source_ref_key`` — a
  re-mentioned entity's live row carries only the latest document's key.
- **dataset**: tombstone every live row in the dataset's namespace, whichever
  delete path ran (the graph-provenance path reports no element list).

Only rows that exist and are live are invalidated (``invalidate`` raises on
untracked ids, and the ledger is opt-in, so most deletes find nothing).
Tombstones are committed in chained batches, and like every other ledger
write path this never raises: deletion must not fail because of the audit
sidecar.
"""

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.provenance import get_data_id_from_source_ref_key
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


def dataset_id_from_ledger_key(entity_id: str) -> UUID | None:
    """The dataset that owns a ledger key, or None for an unscoped key.

    Inverse of ``ledger_node_key`` / ``ledger_edge_key`` — the read surface
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
    """user.email > str(user.id) > "cognee" — the writer's attribution rule."""
    if user is not None:
        email = getattr(user, "email", None)
        if email:
            return str(email)
        user_id = getattr(user, "id", None)
        if user_id is not None:
            return str(user_id)
    return "cognee"


async def _tombstone_ids(
    entity_ids: Iterable[str],
    *,
    agent_id: str,
    reason: str,
    metadata: dict[str, Any] | None,
) -> int:
    manager = get_provenance_manager()
    written = 0
    batch = manager.batch()
    for entity_id in entity_ids:
        batch.invalidate(entity_id, agent_id=agent_id, reason=reason, metadata=metadata)
        if len(batch) >= _COMMIT_CHUNK:
            results = await batch.commit()
            if results is None:
                return written
            written += len(results)
    if len(batch):
        results = await batch.commit()
        if results is None:
            return written
        written += len(results)
    return written


async def tombstone_deleted_elements(
    dataset_id: UUID,
    deleted_elements,
    *,
    user: Any = None,
    data_id: UUID | None = None,
    reason: str = "data_deleted",
) -> int:
    """Tombstone the ledger rows of elements a data-item delete hard-deleted.

    ``deleted_elements`` is a ``DeletedGraphElements``. Returns the number of
    tombstones written (0 when the ledger holds nothing for them, or on any
    failure — never raises).
    """
    if deleted_elements is None:
        return 0
    try:
        keys = [ledger_node_key(dataset_id, node_id) for node_id in deleted_elements.node_ids]
        keys += [
            ledger_edge_key(dataset_id, source, target, relationship_name)
            for source, target, relationship_name in getattr(deleted_elements, "edge_keys", ())
        ]
        if not keys:
            return 0
        live_ids = await storage.retrieve_live_ids(sorted(keys))
        if not live_ids:
            return 0
        metadata = {"deleted_dataset_id": str(dataset_id)}
        if data_id is not None:
            metadata["deleted_data_id"] = str(data_id)
        written = await _tombstone_ids(
            live_ids, agent_id=agent_id_for(user), reason=reason, metadata=metadata
        )
        logger.info(
            "Tombstoned %d ledger entries for data %s in dataset %s", written, data_id, dataset_id
        )
        return written
    except Exception as error:
        logger.warning(
            "Ledger tombstoning failed for data %s in dataset %s (non-fatal): %s",
            data_id,
            dataset_id,
            error,
            exc_info=True,
        )
        return 0


async def tombstone_pipeline_run(
    dataset_id: UUID,
    pipeline_run_id: UUID | str,
    *,
    user: Any = None,
    keep_data_ids: Iterable[UUID | str] | None = None,
    reason: str = "pipeline_run_rolled_back",
) -> int:
    """Tombstone what a rolled-back cognify run first asserted.

    Rows are matched by ``bundle_id`` (the run id) inside the dataset's
    namespace. Only rows with no earlier version are tombstoned: a rollback
    removes the run's ownership refs and hard-deletes artifacts left unowned,
    so an entity the run merely re-mentioned (``previous_version_id`` set)
    survives in the graph and stays live here too — its chain still shows the
    run's version, which is what happened. Rows attributed to documents in
    ``keep_data_ids`` (completed documents a recovery rollback keeps) are left
    alone. Never raises.
    """
    try:
        kept = {str(data_id) for data_id in (keep_data_ids or ())}
        pending: list[str] = []
        for prefix in (f"{dataset_id}:", f"rel:{dataset_id}:"):
            async for entry in storage.iter_live_by_bundle(str(pipeline_run_id), prefix):
                if entry.previous_version_id:
                    continue
                if kept and entry.source_ref_key:
                    try:
                        data_id = str(get_data_id_from_source_ref_key(entry.source_ref_key))
                    except ValueError:
                        data_id = None
                    if data_id in kept:
                        continue
                pending.append(entry.entity_id)
        if not pending:
            return 0
        written = await _tombstone_ids(
            pending,
            agent_id=agent_id_for(user),
            reason=reason,
            metadata={
                "deleted_dataset_id": str(dataset_id),
                "rolled_back_pipeline_run_id": str(pipeline_run_id),
            },
        )
        logger.info(
            "Tombstoned %d ledger entries for rolled-back run %s (dataset %s)",
            written,
            pipeline_run_id,
            dataset_id,
        )
        return written
    except Exception as error:
        logger.warning(
            "Ledger tombstoning failed for rolled-back run %s (non-fatal): %s",
            pipeline_run_id,
            error,
            exc_info=True,
        )
        return 0


async def tombstone_dataset(
    dataset_id: UUID,
    *,
    user: Any = None,
    reason: str = "dataset_deleted",
) -> int:
    """Tombstone every live ledger row in the dataset's namespace.

    Covers nodes (``"{dataset_id}:..."``) and relationships
    (``"rel:{dataset_id}:..."``). Returns the number of tombstones written;
    never raises.
    """
    try:
        metadata = {"deleted_dataset_id": str(dataset_id)}
        agent_id = agent_id_for(user)
        written = 0
        for prefix in (f"{dataset_id}:", f"rel:{dataset_id}:"):
            pending: list[str] = []
            async for entity_id in storage.iter_live_with_prefix(prefix):
                pending.append(entity_id)
            # Collected before writing: the prefix scan and the tombstone
            # writes must not interleave on one ledger (the scan session
            # would otherwise observe its own in-flight invalidations).
            if pending:
                written += await _tombstone_ids(
                    pending, agent_id=agent_id, reason=reason, metadata=metadata
                )
        if written:
            logger.info("Tombstoned %d ledger entries for dataset %s", written, dataset_id)
        return written
    except Exception as error:
        logger.warning(
            "Ledger tombstoning failed for dataset %s (non-fatal): %s",
            dataset_id,
            error,
            exc_info=True,
        )
        return 0
