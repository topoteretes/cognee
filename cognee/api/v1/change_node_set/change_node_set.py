import json
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from cognee.api.v1.exceptions import UpdateTargetNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.data.methods import get_authorized_dataset, get_data, resolve_data_id
from cognee.modules.data.models import Data
from cognee.modules.engine.models.node_set import (
    InvalidNodeSetError,
    NodeSet,
    parse_node_set_names,
    validate_node_set_names,
)
from cognee.modules.graph.methods.delete_chunks_incremental import edge_endpoints
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("change_node_set")

CHUNK_COLLECTION = "DocumentChunk_text"
BELONGS_TO_SET = "belongs_to_set"


async def change_node_set(
    data_id: UUID,
    dataset_id: UUID,
    node_set: list[str],
    user: User | None = None,
) -> dict:
    """Move a document already in memory into a different node set, in place.

    ``add(node_set=["alpha"])`` then ``cognify()`` tags a document and its chunks
    with ``alpha``. Reorganising memory later — moving the document to ``beta`` —
    went through ``update()``, which drops the document's memory and re-adds it, so
    the next cognify re-runs the full LLM extraction, re-chunks and re-embeds. That
    cost is wasted when only the label changed: the content, the chunk boundaries
    and the vectors are all identical.

    ``change_node_set`` rewrites only the node-set membership. It REPLACES the
    document's node set (a move, not an add): after the call the content belongs to
    ``node_set`` everywhere membership is read and to none of the previous sets.
    Three stores are touched, none of which re-embeds or calls an LLM:

    - Vector: each chunk's ``belongs_to_set`` / ``source_node_set`` payload fields
      are overwritten via ``update_payload`` — the stored vector and the chunk id
      are preserved exactly, so a ``CHUNKS`` search (which filters on the chunk
      payload ``belongs_to_set``) finds the chunks under the new set and no longer
      under the old one.
    - Graph: the document node and its chunk nodes get the removed sets' tags and
      ``belongs_to_set`` edges stripped (scoped to these node ids, so sets shared
      with other documents are left alone), the new sets' ``NodeSet`` nodes and
      edges added, and their ``belongs_to_set`` / ``source_node_set`` properties set
      to exactly the new set.
    - Relational: ``Data.node_set`` and the ``node_set`` key of
      ``Data.external_metadata`` both move, so a later full ``cognify()`` rebuilds
      the NodeSet tags from the new value rather than the one stored at ingestion.

    Args:
        data_id: UUID of the document to move (current id or a pre-fork legacy id).
        dataset_id: UUID of the dataset holding the document (required for a precise,
            unambiguous id resolution and for the write-permission check).
        node_set: The names the document should belong to after the call. A
            non-empty list replaces the current set; ``[]`` clears membership
            entirely. Validated as ``add``/``update`` validate it — a list of
            non-empty strings, never a bare string.
        user: User performing the change. Defaults to the default user. The caller
            must have ``write`` access to the dataset.

    Returns:
        A summary dict:
            - ``status``: ``"retagged"``.
            - ``data_id``, ``dataset_id``: the document moved and its dataset.
            - ``node_set``: the set the document now belongs to.
            - ``previous_node_set``: the set it belonged to before the call.
            - ``chunks_retagged``: how many chunk vectors were repointed.
            - ``duration_seconds``: wall-clock time of the change.

    Raises:
        InvalidNodeSetError: ``node_set`` is not a list of names.
        PermissionDeniedError: the user cannot write to the dataset.
        UpdateTargetNotFoundError: ``data_id`` resolves to no document in the dataset.
    """
    started = perf_counter()

    # Reject a malformed node set before any store is touched; a move with an
    # unusable target would otherwise strip the old set and then fail to apply
    # the new one, leaving the document in neither. None is a misuse here (the
    # parameter is required): pass [] to clear membership, a list to replace it.
    if node_set is None:
        raise InvalidNodeSetError(
            "change_node_set requires a list of names (use [] to clear), got None"
        )
    validate_node_set_names(node_set)
    new_names = list(node_set)

    if not user:
        user = await get_default_user()

    # Authorize before the id is resolved, like update(): a caller without write
    # access must not learn from the 404-versus-further response whether a
    # (dataset, data_id) pair exists.
    if await get_authorized_dataset(user, dataset_id, "write") is None:
        raise PermissionDeniedError(message="Not authorized to change node set in this dataset")

    resolved_id = await resolve_data_id(dataset_id, data_id)
    if resolved_id is None:
        raise UpdateTargetNotFoundError(data_id=data_id, dataset_id=dataset_id)
    document_id = resolved_id

    # The dataset ACL already authorized the move, so row ownership is not the
    # access rule here (a collaborator may move a document they do not own).
    data = await get_data(user.id, document_id, dataset_id, verify_owner=False)
    old_names = parse_node_set_names(data.node_set) if data is not None else []

    graph_engine = await get_graph_engine()
    chunk_ids = await _document_chunk_ids(graph_engine, document_id)
    node_ids = [str(document_id), *chunk_ids]

    await _retag_graph_nodes(graph_engine, node_ids, old_names, new_names)

    # Only chunks are embedded; the document node lives in the graph alone, so the
    # vector rewrite is the chunk payloads. Skipped when the document has no stored
    # chunks (never cognified) — the relational move below still lands so a later
    # cognify tags it with the new set.
    if chunk_ids:
        vector_engine = await get_vector_engine_async()
        source_node_set = ", ".join(new_names)
        await vector_engine.update_payload(
            CHUNK_COLLECTION,
            {
                chunk_id: {
                    BELONGS_TO_SET: new_names,
                    "source_node_set": source_node_set,
                }
                for chunk_id in chunk_ids
            },
        )

    await _publish_node_set(document_id, new_names)

    logger.info(
        "moved document %s from node_set %s to %s (%d chunks)",
        document_id,
        old_names,
        new_names,
        len(chunk_ids),
    )

    return {
        "status": "retagged",
        "data_id": str(document_id),
        "dataset_id": str(dataset_id),
        "node_set": new_names,
        "previous_node_set": old_names,
        "chunks_retagged": len(chunk_ids),
        "duration_seconds": round(perf_counter() - started, 3),
    }


async def _document_chunk_ids(graph_engine, document_id: UUID) -> list[str]:
    """The ids of the document's chunk nodes, found via their ``is_part_of`` edges.

    Mirrors how the incremental update discovers a document's chunks: adapters
    disagree on connection shapes, so the true endpoints are resolved with
    ``edge_endpoints`` before an edge is read as chunk -> document.
    """
    connections = await graph_engine.get_connections(str(document_id))
    chunk_ids: list[str] = []
    for source, edge, target in connections:
        if "is_part_of" not in str(edge.get("relationship_name", "")):
            continue
        source_id, target_id = edge_endpoints(source, edge, target)
        if target_id == str(document_id) and source_id != str(document_id):
            chunk_ids.append(source_id)
    return chunk_ids


async def _retag_graph_nodes(
    graph_engine,
    node_ids: list[str],
    old_names: list[str],
    new_names: list[str],
) -> None:
    """Repoint the graph node-set membership of ``node_ids`` to exactly ``new_names``.

    Edges and the tag property are reconciled separately because the engine keeps
    both: a ``belongs_to_set`` property array on each node (what the vector filter
    and the graph read) and additive ``belongs_to_set`` edges to ``NodeSet`` nodes
    (what a graph-scoped NodeSet projection walks).
    """
    source_node_set = ", ".join(new_names)
    to_remove = [name for name in old_names if name not in new_names]

    # Drop the edges and property tags for sets the document is leaving, scoped to
    # these node ids so a set shared with another document keeps its other members.
    if to_remove:
        await graph_engine.remove_belongs_to_set_tags(to_remove, node_ids=node_ids)

    # Add the new sets: ensure each NodeSet node exists, then an idempotent
    # belongs_to_set edge from every retagged node to it.
    for name in new_names:
        node_set_id = str(NodeSet.id_for(name))
        await graph_engine.add_node(NodeSet(name=name))
        for node_id in node_ids:
            await graph_engine.add_edge(
                node_id,
                node_set_id,
                "belongs_to_set",
                properties={
                    "source_node_id": node_id,
                    "target_node_id": node_set_id,
                    "relationship_name": "belongs_to_set",
                    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    # Set the tag property to exactly the new set. This is the authoritative write
    # for the property array: the removal above only strips the leaving sets and
    # the edge adds do not touch the property, so this is what guarantees a node
    # left with precisely new_names regardless of what it carried before.
    for node_id in node_ids:
        await graph_engine.update_node(
            node_id,
            {BELONGS_TO_SET: new_names, "source_node_set": source_node_set},
        )


async def _publish_node_set(document_id: UUID, new_names: list[str]) -> None:
    """Move the relational node set: the column readers filter on and the metadata
    a later full cognify rebuilds its NodeSet tags from both change together."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == document_id))
        ).scalar_one_or_none()
        if data_point is None:
            raise ValueError(f"data row {document_id} disappeared before the node-set change")

        data_point.node_set = json.dumps(new_names) if new_names else None

        metadata = dict(data_point.external_metadata or {})
        if new_names:
            metadata["node_set"] = new_names
        else:
            metadata.pop("node_set", None)
        data_point.external_metadata = metadata
        # The JSON column is mutated in place, so SQLAlchemy must be told it
        # changed or the move would not be flushed.
        flag_modified(data_point, "external_metadata")

        await session.merge(data_point)
        await session.commit()
