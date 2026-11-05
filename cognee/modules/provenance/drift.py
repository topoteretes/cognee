"""Ledger-vs-graph drift: has a node changed since the ledger last recorded it?

The hash chain proves the *ledger* was not altered. It says nothing about
the graph: an out-of-band edit to a node (a Cypher write, a migration, a
backend restore) leaves the ledger internally valid and silently wrong.
``check_drift`` closes that gap using the snapshot every row carries
(``snapshot.py``): each live node row of a dataset is read back from the
graph, re-projected onto the fields the snapshot recorded, re-hashed and
compared.

Findings:

- ``drifted``: the node exists but its content hash differs; the field
  level difference is reported.
- ``missing_in_graph``: the ledger holds a live row but the graph has no
  such node — deleted without a tombstone.
- ``unsnapshotted``: live rows written before snapshots existed; counted,
  not checked.

Relationships are not checked (their ledger identity *is* their content).
The graph is read under the dataset's database context, so this works on
isolated per-dataset databases; the caller resolves the dataset and its
owner (and enforces ACL) before calling.
"""

from typing import Any
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger

from . import storage
from .snapshot import content_hash, diff_snapshots, snapshot_from_mapping

logger = get_logger("provenance.drift")

_NODE_KINDS = frozenset({"entity", "document", "chunk"})
_PAGE = 500


async def check_drift(dataset_id: UUID | str, owner_id: UUID | None = None) -> dict[str, Any]:
    prefix = f"{dataset_id}:"
    checked = 0
    unsnapshotted = 0
    drifted: list[dict[str, Any]] = []
    missing: list[str] = []

    async def compare(page: list[Any], graph: Any) -> None:
        nonlocal checked
        raw_ids = [entry.entity_id[len(prefix) :] for entry in page]
        nodes = await graph.get_nodes(raw_ids)
        by_id = {str(node.get("id")): node for node in nodes if isinstance(node, dict)}
        for entry, raw_id in zip(page, raw_ids):
            checked += 1
            snapshot = entry.metadata["snapshot"]
            node = by_id.get(raw_id)
            if node is None:
                missing.append(entry.entity_id)
                continue
            fields = snapshot_from_mapping(node, snapshot.get("fields") or {})
            observed = content_hash(fields)
            if observed != snapshot.get("hash"):
                drifted.append(
                    {
                        "entity_id": entry.entity_id,
                        "entity_type": entry.entity_type,
                        "recorded_hash": snapshot.get("hash"),
                        "observed_hash": observed,
                        "delta": diff_snapshots(snapshot.get("fields"), fields),
                    }
                )

    async with set_database_global_context_variables(
        UUID(str(dataset_id)), UUID(str(owner_id)) if owner_id else None
    ):
        graph = await get_graph_engine()
        page: list[Any] = []
        async for entry in storage.iter_live_entries(prefix):
            if entry.entity_type not in _NODE_KINDS:
                continue
            snapshot = entry.metadata.get("snapshot") if isinstance(entry.metadata, dict) else None
            if not isinstance(snapshot, dict) or not snapshot.get("hash"):
                unsnapshotted += 1
                continue
            page.append(entry)
            if len(page) >= _PAGE:
                await compare(page, graph)
                page = []
        if page:
            await compare(page, graph)

    return {
        "valid": not drifted and not missing,
        "dataset_id": str(dataset_id),
        "checked": checked,
        "unsnapshotted": unsnapshotted,
        "drifted": drifted,
        "missing_in_graph": missing,
    }
