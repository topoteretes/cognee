"""Opt-in cognify task writing the audit-grade provenance ledger.

Pipeline seam: spliced in by ``get_default_tasks`` right after
``add_data_points`` (node ids are persisted and stable) and before the
contradiction-detection spread, when the ``provenance_tracking`` CognifyConfig
flag is on (env ``PROVENANCE_TRACKING``, default off).

For every item this ingestion produced it appends document -> chunk -> entity
-> relationship lineage entries to the relational ``provenance_entries``
ledger, joined to graph provenance by the existing source-ref key. All entries
for one task invocation are committed as ONE chained transaction
(``manager.batch()``), matching the pipeline's batching convention.

Tenant/dataset scoping: cognee entity ids are globally deterministic
(name-derived uuid5s), and the ledger lives in the always-shared relational
DB — so every ledger key is prefixed with the dataset id from ctx
(``"{dataset_id}:{raw_id}"``). Without this, two tenants mentioning the same
entity name would share (and cross-overwrite) one version chain. Readers must
join with the same prefixed key; entries written without ctx stay unprefixed.

Custom ``graph_model`` pipelines whose DataPoints lack the default
made_from/is_part_of/contains shape are covered by a generic fallback that
walks the model with ``get_graph_from_model`` (the same traversal
``add_data_points`` uses) and records every node and edge.

Hard constraints, mirroring ``detect_contradictions``: the task returns its
input unchanged and swallows all of its own errors — provenance can never
break ingestion. Missing ctx, missing dataset/data ids, or raw items with no
document degrade to entries with ``source_ref_key=None``, never a raise.
"""

from typing import List, Optional

from cognee.infrastructure.databases.provenance import data_item_id, make_source_ref_key
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.modules.provenance import get_provenance_manager
from cognee.shared.logging_utils import get_logger

logger = get_logger("record_provenance")


def _agent_from_ctx(ctx: Optional[PipelineContext]) -> str:
    """user.email > str(user.id) > "cognee" (mirrors source_user stamping rules)."""
    user = getattr(ctx, "user", None)
    if user is not None:
        email = getattr(user, "email", None)
        if email:
            return str(email)
        user_id = getattr(user, "id", None)
        if user_id is not None:
            return str(user_id)
    return "cognee"


def _source_ref_key_from_ctx(ctx: Optional[PipelineContext]) -> Optional[str]:
    """Reuse the graph-provenance source-ref key; None when either id is missing."""
    if ctx is None:
        return None
    dataset = getattr(ctx, "dataset", None)
    dataset_id = getattr(dataset, "id", None) if dataset is not None else None
    data_id = data_item_id(getattr(ctx, "data_item", None))
    if dataset_id and data_id:
        return make_source_ref_key(dataset_id, data_id)
    return None


def _scope_from_ctx(ctx: Optional[PipelineContext]) -> Optional[str]:
    """Dataset id as the ledger namespace (per-tenant/dataset isolation)."""
    dataset = getattr(ctx, "dataset", None) if ctx is not None else None
    dataset_id = getattr(dataset, "id", None) if dataset is not None else None
    return str(dataset_id) if dataset_id else None


def _entity_type_name(entity) -> Optional[str]:
    is_a = getattr(entity, "is_a", None)
    name = getattr(is_a, "name", None) if is_a is not None else None
    return name or type(entity).__name__


def _relationship_name(edge, default: str) -> str:
    return getattr(edge, "relationship_type", None) or default


def _has_default_shape(item, chunk) -> bool:
    """True when the item follows the default cognify schema walk."""
    return (
        getattr(item, "made_from", None) is not None
        or getattr(chunk, "is_part_of", None) is not None
        or getattr(chunk, "contains", None) is not None
    )


@task_summary("Recorded provenance for {n} item(s)")
async def record_provenance(
    data_points: List[DataPoint], ctx: Optional[PipelineContext] = None
) -> List[DataPoint]:
    """Append ledger entries for everything this ingestion produced.

    Receives the batch ``add_data_points`` returns: ``TextSummary`` items
    wrapping chunks via ``made_from``, ``DocumentChunk`` objects directly, or
    custom-schema DataPoints (generic traversal). Returns the input unchanged
    so it can be appended to any pipeline.
    """
    if not isinstance(data_points, list) or not data_points:
        return data_points

    try:
        manager = get_provenance_manager()

        run_id = str(ctx.pipeline_run_id) if ctx and ctx.pipeline_run_id else None
        common = {
            "activity_id": f"cognify:{run_id}" if run_id else "cognify",
            "agent_id": _agent_from_ctx(ctx),
            "bundle_id": run_id,
            "source_ref_key": _source_ref_key_from_ctx(ctx),
        }
        scope = _scope_from_ctx(ctx)

        def scoped(raw_id: str) -> str:
            return f"{scope}:{raw_id}" if scope else raw_id

        batch = manager.batch()

        # Dedup state is per invocation, but the task runs once per chunk
        # batch — for documents spanning batches the ledger itself is
        # consulted: an entry already carrying THIS run's bundle_id was
        # written by an earlier batch of the same ingestion, so re-tracking
        # (which would forge a spurious version) is skipped.
        tracked_documents = set()

        async def should_track_document(ledger_id: str) -> bool:
            if ledger_id in tracked_documents:
                return False
            tracked_documents.add(ledger_id)
            if common["bundle_id"]:
                existing = await manager.get_provenance(ledger_id)
                if existing and existing.get("bundle_id") == common["bundle_id"]:
                    return False
            return True

        # Shared traversal state for the generic (custom-schema) path.
        generic_added_nodes: dict = {}
        generic_added_edges: dict = {}
        generic_visited: dict = {}

        for item in data_points:
            chunk = getattr(item, "made_from", None) or item

            if not _has_default_shape(item, chunk) and isinstance(item, DataPoint):
                # Custom graph_model schema: no made_from/is_part_of/contains.
                # Walk the model generically so the audit still covers every
                # node and edge this ingestion produced.
                nodes, edges = await get_graph_from_model(
                    item,
                    added_nodes=generic_added_nodes,
                    added_edges=generic_added_edges,
                    visited_properties=generic_visited,
                )
                root_id = str(item.id)
                # The walk may legitimately return a node set that does not include
                # ``item``. Attribute to it only when it was actually stored.
                root_source = (
                    scoped(root_id) if any(str(node.id) == root_id for node in nodes) else ""
                )
                for node in nodes:
                    node_id = str(node.id)
                    batch.track_entity(
                        scoped(node_id),
                        source="" if node_id == root_id else root_source,
                        entity_type="entity",
                        metadata={
                            "name": getattr(node, "name", None),
                            "type": getattr(node, "type", type(node).__name__),
                        },
                        **common,
                    )
                for edge_source_id, edge_target_id, relationship_name, _ in edges:
                    source_id = scoped(str(edge_source_id))
                    target_id = scoped(str(edge_target_id))
                    batch.track_relationship(
                        f"rel:{source_id}:{relationship_name}:{target_id}",
                        source=root_source,
                        used_entities=[source_id, target_id],
                        metadata={"relationship_name": relationship_name},
                        **common,
                    )
                continue

            document = getattr(chunk, "is_part_of", None)

            document_id = None
            if document is not None and getattr(document, "id", None) is not None:
                document_id = scoped(str(document.id))
                if await should_track_document(document_id):
                    batch.track_entity(
                        document_id,
                        source=getattr(document, "raw_data_location", None)
                        or getattr(document, "name", None)
                        or "",
                        entity_type="document",
                        **common,
                    )

            chunk_id = getattr(chunk, "id", None)
            chunk_id = scoped(str(chunk_id)) if chunk_id is not None else None
            if chunk_id is not None:
                chunk_index = getattr(chunk, "chunk_index", 0) or 0
                batch.track_chunk(
                    chunk_id,
                    source_document=(
                        getattr(document, "raw_data_location", None) or "" if document else ""
                    ),
                    parent_chunk_id=document_id,
                    start_index=chunk_index,
                    end_index=chunk_index,
                    chunk_size=getattr(chunk, "chunk_size", None),
                    **common,
                )

            for entry in getattr(chunk, "contains", None) or []:
                # ``contains`` entries are Entity/Event nodes or (Edge, node) tuples.
                edge = entry[0] if isinstance(entry, tuple) else None
                entity = entry[1] if isinstance(entry, tuple) else entry
                entity_id = getattr(entity, "id", None)
                if entity_id is None:
                    continue
                entity_id = scoped(str(entity_id))

                # Re-mention across documents auto-versions: entity ids are
                # name-derived uuid5s (deterministic within the dataset scope
                # applied above), so the version chain is exactly the audit
                # record of every mention within one dataset.
                batch.track_entity(
                    entity_id,
                    source=chunk_id or "",
                    entity_type="entity",
                    metadata={
                        "name": getattr(entity, "name", None),
                        "type": _entity_type_name(entity),
                    },
                    **common,
                )

                relationship_pairs = []
                if edge is not None and chunk_id is not None:
                    relationship_pairs.append(
                        (chunk_id, _relationship_name(edge, "contains"), entity_id)
                    )
                for relation in getattr(entity, "relations", None) or []:
                    if not isinstance(relation, tuple) or len(relation) != 2:
                        continue
                    relation_edge, target = relation
                    target_id = getattr(target, "id", None)
                    if target_id is None:
                        continue
                    relationship_pairs.append(
                        (
                            entity_id,
                            _relationship_name(relation_edge, "related_to"),
                            scoped(str(target_id)),
                        )
                    )

                for source_id, relationship_name, target_id in relationship_pairs:
                    batch.track_relationship(
                        f"rel:{source_id}:{relationship_name}:{target_id}",
                        source=chunk_id or "",
                        used_entities=[source_id, target_id],
                        metadata={"relationship_name": relationship_name},
                        **common,
                    )

        await batch.commit()
    except Exception as error:
        logger.warning("Provenance recording failed; ingestion unaffected: %s", error)

    return data_points
