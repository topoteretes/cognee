"""Ledgered-inverse capture and restore for destructive graph/vector operations.

Approach 1 (run-ledger time-travel) keeps ``forget`` and ``rollback`` reversible
without copying the graph: before a destructive source-ref removal runs, this
module captures the *exact inverse* — the graph rows (properties + all four
provenance columns) and the raw vector rows (id + embedding + payload) the
operation will destroy — into a JSON payload stored in the ``version_ops``
write-ahead ledger. Undo replays that payload.

Design constraints this encodes:

- **No cross-store atomicity.** Graph, vector, and relational stores cannot
  share a transaction. The guarantee is write-ahead order (inverse committed
  to the relational ledger *before* the destructive op) plus an idempotent
  restore: every restore primitive is a MERGE upsert or a set-merge attach, so
  a crash mid-restore is fixed by replaying.
- **Model A provenance.** A ``source_run_ref`` exists only for the run that
  *first* attached a key to an artifact, so each key maps to at most one run.
  Restore re-attaches keys grouped by their original owning run, which
  reproduces the provenance columns exactly (including ``source_run_ids`` and
  ``source_dataset_ids``, which the adapters re-derive).
- **Restore timestamps are not original.** Graph-storage ``created_at`` /
  ``updated_at`` columns reflect restore time; artifact identity, properties,
  provenance, and vectors are exact.

Known limitation (documented, not silent): ``belongs_to_set`` tags stripped
from *surviving* artifacts by the planner's orphaned-NodeSet cleanup are not
re-added on undo. The NodeSet node itself and everything it owned are restored.
"""

from typing import Any, Dict, List, Optional

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.databases.provenance.source_refs import (
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
)

# The capture MUST partition deleted-vs-detached with the delete planner's own
# ownership rule — a mirrored copy could drift and make undo silently lossy.
from cognee.infrastructure.databases.unified.provenance_delete_planner import _is_unowned
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.shared.logging_utils import get_logger

logger = get_logger("versioning.inverse")

INVERSE_PAYLOAD_VERSION = 1


class _RestoredArtifact:
    """Property bag whose ``vars()`` is the captured flat node payload.

    ``add_nodes`` accepts any object: it reads ``model_dump()`` when present,
    else ``vars()``. The captured ``node_properties`` is exactly the flat
    payload the adapter reconstructs on read (core columns merged over the
    JSON blob), so round-tripping it through ``vars()`` recreates the node
    byte-equal up to JSON key order.
    """

    def __init__(self, properties: Dict[str, Any]):
        self.__dict__.update(properties)


def _edge_key(source_id: str, target_id: str, relationship_name: str) -> EdgeIdentity:
    return EdgeIdentity(
        source_id=source_id, target_id=target_id, relationship_name=relationship_name
    )


def _group_keys_by_owning_run(
    source_ref_keys: List[str], source_run_refs: List[str]
) -> Dict[Optional[str], List[str]]:
    """Group ref keys by the run that first attached them (None = no run ref)."""
    run_by_key: Dict[str, str] = {}
    for run_ref in source_run_refs:
        key = get_source_ref_key_from_source_run_ref(run_ref)
        run_by_key[key] = str(get_pipeline_run_id_from_source_run_ref(run_ref))

    groups: Dict[Optional[str], List[str]] = {}
    for key in source_ref_keys:
        groups.setdefault(run_by_key.get(key), []).append(key)
    return groups


async def _capture_vector_rows(
    vector_engine, collection: str, ids: List[str], vector_rows: Dict[str, List[dict]]
) -> None:
    if not ids:
        return
    rows = await vector_engine.get_raw_rows(collection, ids)
    if rows:
        vector_rows.setdefault(collection, []).extend(rows)


async def capture_source_ref_removal_inverse(
    graph_engine,
    vector_engine,
    *,
    refs_by_node: Dict[str, List[str]],
    refs_by_edge: Dict[EdgeIdentity, List[str]],
) -> Dict[str, Any]:
    """Capture the exact inverse of one planned source-ref removal.

    ``refs_by_node`` / ``refs_by_edge`` are the same inputs the delete planner
    receives (artifact -> refs about to be removed), so the capture partitions
    artifacts into deleted-vs-detached with the identical ownership rule and
    snapshots everything the planner will destroy: full graph rows and
    provenance for unowned artifacts, the removed refs (with their original
    run refs) for surviving ones, and the raw vector rows for every vector
    point the planner deletes (node index fields, Triplet_text, and the
    EdgeType rows its orphan cleanup may prune).
    """
    node_data = await graph_engine.get_node_delete_data(list(refs_by_node.keys()))
    edge_data = await graph_engine.get_edge_delete_data(list(refs_by_edge.keys()))

    nodes_deleted: List[dict] = []
    nodes_detached: List[dict] = []
    edges_deleted: List[dict] = []
    edges_detached: List[dict] = []
    vector_rows: Dict[str, List[dict]] = {}

    unowned_node_ids: List[str] = []
    for node_id, data in node_data.items():
        removed = refs_by_node.get(node_id, [])
        if _is_unowned(data.source_ref_keys, removed):
            unowned_node_ids.append(node_id)
            nodes_deleted.append(
                {
                    "node_id": data.node_id,
                    "node_type": data.node_type,
                    "indexed_fields": list(data.indexed_fields),
                    "node_properties": dict(data.node_properties),
                    "source_ref_keys": list(data.source_ref_keys),
                    "source_run_refs": list(data.source_run_refs),
                }
            )
        else:
            removed_set = set(removed)
            nodes_detached.append(
                {
                    "node_id": data.node_id,
                    "removed_ref_keys": list(removed),
                    "removed_run_refs": [
                        run_ref
                        for run_ref in data.source_run_refs
                        if get_source_ref_key_from_source_run_ref(run_ref) in removed_set
                    ],
                }
            )

    unowned_edges: List[EdgeIdentity] = []
    for edge, data in edge_data.items():
        removed = refs_by_edge.get(edge, [])
        if _is_unowned(data.source_ref_keys, removed):
            unowned_edges.append(edge)
            edges_deleted.append(
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_name": edge.relationship_name,
                    "edge_properties": dict(data.edge_properties),
                    "source_ref_keys": list(data.source_ref_keys),
                    "source_run_refs": list(data.source_run_refs),
                }
            )
        else:
            removed_set = set(removed)
            edges_detached.append(
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_name": edge.relationship_name,
                    "removed_ref_keys": list(removed),
                    "removed_run_refs": [
                        run_ref
                        for run_ref in data.source_run_refs
                        if get_source_ref_key_from_source_run_ref(run_ref) in removed_set
                    ],
                }
            )

    # Vector rows the planner will delete: per-node index-field collections.
    node_collections: Dict[str, List[str]] = {}
    for node_id in unowned_node_ids:
        data = node_data[node_id]
        for field in data.indexed_fields:
            node_collections.setdefault(f"{data.node_type}_{field}", []).append(node_id)
    for collection, ids in node_collections.items():
        await _capture_vector_rows(vector_engine, collection, ids, vector_rows)

    # Per-edge triplet vectors + the EdgeType artifacts the orphan cleanup may
    # prune. EdgeType restore is a MERGE keyed by relationship text, so
    # capturing the superset (all texts touched, orphaned or not) stays exact.
    edge_type_nodes: List[dict] = []
    if unowned_edges:
        triplet_ids = [
            str(generate_node_id(edge.source_id + edge.relationship_name + edge.target_id))
            for edge in unowned_edges
        ]
        await _capture_vector_rows(vector_engine, "Triplet_text", triplet_ids, vector_rows)

        edge_texts = {
            edge_data[edge].edge_text for edge in unowned_edges if edge_data[edge].edge_text
        }
        edge_type_ids = [str(EdgeType.id_for(text)) for text in sorted(edge_texts)]
        if edge_type_ids:
            edge_type_data = await graph_engine.get_node_delete_data(edge_type_ids)
            for node_id, data in edge_type_data.items():
                edge_type_nodes.append(
                    {
                        "node_id": data.node_id,
                        "node_type": data.node_type,
                        "node_properties": dict(data.node_properties),
                    }
                )
            await _capture_vector_rows(
                vector_engine, "EdgeType_relationship_name", edge_type_ids, vector_rows
            )

    return {
        "nodes_deleted": nodes_deleted,
        "nodes_detached": nodes_detached,
        "edges_deleted": edges_deleted,
        "edges_detached": edges_detached,
        "edge_type_nodes": edge_type_nodes,
        "vector_rows": vector_rows,
    }


async def _attach_grouped_node_refs(graph_engine, groups: Dict[tuple, List[str]]) -> None:
    for (run_id, keys), node_ids in groups.items():
        await graph_engine.attach_node_source_refs(node_ids, list(keys), run_id)


async def _attach_grouped_edge_refs(graph_engine, groups: Dict[tuple, List[EdgeIdentity]]) -> None:
    for (run_id, keys), edges in groups.items():
        await graph_engine.attach_edge_source_refs(edges, list(keys), run_id)


async def restore_inverse_step(graph_engine, vector_engine, step: Dict[str, Any]) -> None:
    """Replay one captured inverse step. Idempotent: safe to re-run after a crash.

    Order matters only where restores depend on each other: nodes before edges
    (edge MERGE matches endpoints), artifacts before provenance attach.
    """
    nodes_deleted = step.get("nodes_deleted", [])
    edges_deleted = step.get("edges_deleted", [])

    # 1. Recreate deleted nodes (MERGE by id) and the EdgeType nodes the
    #    orphan cleanup may have pruned.
    restored_nodes = [_RestoredArtifact(dict(node["node_properties"])) for node in nodes_deleted]
    restored_nodes.extend(
        _RestoredArtifact(dict(node["node_properties"])) for node in step.get("edge_type_nodes", [])
    )
    if restored_nodes:
        await graph_engine.add_nodes(restored_nodes)

    # 2. Recreate deleted edges (MERGE by endpoints + relationship_name).
    if edges_deleted:
        await graph_engine.add_edges(
            [
                (
                    edge["source_id"],
                    edge["target_id"],
                    edge["relationship_name"],
                    dict(edge["edge_properties"]),
                )
                for edge in edges_deleted
            ]
        )

    # 3. Re-attach provenance, grouped by the original owning run. Recreated
    #    artifacts start with empty provenance, so every key is "newly
    #    attached" and Model A records exactly the original run refs.
    node_groups: Dict[tuple, List[str]] = {}
    for node in nodes_deleted:
        for run_id, keys in _group_keys_by_owning_run(
            node["source_ref_keys"], node["source_run_refs"]
        ).items():
            node_groups.setdefault((run_id, tuple(keys)), []).append(node["node_id"])
    for node in step.get("nodes_detached", []):
        for run_id, keys in _group_keys_by_owning_run(
            node["removed_ref_keys"], node["removed_run_refs"]
        ).items():
            node_groups.setdefault((run_id, tuple(keys)), []).append(node["node_id"])
    await _attach_grouped_node_refs(graph_engine, node_groups)

    edge_groups: Dict[tuple, List[EdgeIdentity]] = {}
    for edge in edges_deleted:
        for run_id, keys in _group_keys_by_owning_run(
            edge["source_ref_keys"], edge["source_run_refs"]
        ).items():
            edge_groups.setdefault((run_id, tuple(keys)), []).append(
                _edge_key(edge["source_id"], edge["target_id"], edge["relationship_name"])
            )
    for edge in step.get("edges_detached", []):
        for run_id, keys in _group_keys_by_owning_run(
            edge["removed_ref_keys"], edge["removed_run_refs"]
        ).items():
            edge_groups.setdefault((run_id, tuple(keys)), []).append(
                _edge_key(edge["source_id"], edge["target_id"], edge["relationship_name"])
            )
    await _attach_grouped_edge_refs(graph_engine, edge_groups)

    # 4. Restore the exact vector rows (id + embedding + payload). No
    #    re-embedding: restored vectors are the captured ones.
    for collection, rows in step.get("vector_rows", {}).items():
        await vector_engine.restore_raw_rows(collection, rows)

    logger.info(
        "Restored inverse step: %d nodes, %d edges, %d detached nodes, %d detached edges, "
        "%d vector collections.",
        len(nodes_deleted),
        len(edges_deleted),
        len(step.get("nodes_detached", [])),
        len(step.get("edges_detached", [])),
        len(step.get("vector_rows", {})),
    )
