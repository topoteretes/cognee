import copy
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import NAMESPACE_OID, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine_async
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.modules.data.models import Data, Dataset
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.methods.upsert_edges import upsert_edges
from cognee.modules.graph.methods.upsert_nodes import upsert_nodes
from cognee.modules.graph.models import Edge, Node
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("link_data_to_dataset")

LEDGER_BATCH_SIZE = 1000


def _overrides_tag_add(engine, interface_default) -> bool:
    """Best-effort check that `engine` overrides `add_belongs_to_set_tags`.

    Resolution goes through the *instance*, not the class: `get_graph_engine()`
    and `get_vector_engine_async()` return handle objects that forward every
    attribute access to the live adapter via `__getattr__`, so the adapter's
    methods never appear on the handle's class. Bound methods are compared by
    `__func__` identity against the interface default (which raises
    NotImplementedError). A callable without `__func__` — e.g. the leased-cache
    proxy wraps adapter methods in a plain closure — is inconclusive and counts
    as supported; an adapter that then turns out not to implement tagging still
    raises NotImplementedError, which the caller catches to fall back before
    the link is treated as done (all link writes are additive and idempotent,
    so the full-run fallback converges).
    """
    method = getattr(engine, "add_belongs_to_set_tags", None)
    if method is None:
        return False
    func = getattr(method, "__func__", None)
    if func is None:
        return True
    return func is not interface_default


def _supports_tag_add(graph_engine, vector_engine) -> bool:
    """True unless an engine provably lacks `add_belongs_to_set_tags`.

    Detects missing tagging support up front — before any store has been
    mutated — so the caller can fall back to full processing cheaply.
    """
    return _overrides_tag_add(
        graph_engine, GraphDBInterface.add_belongs_to_set_tags
    ) and _overrides_tag_add(vector_engine, VectorDBInterface.add_belongs_to_set_tags)


def _data_node_set(data: Data) -> List[str]:
    """The node_set names the current add supplied for `data`, or []."""
    if not data.node_set:
        return []
    try:
        node_set = json.loads(data.node_set)
    except (TypeError, ValueError):
        return []
    if not isinstance(node_set, list):
        return []
    return [str(name) for name in node_set]


def _copied_node_attributes(row: Node, tags: List[str]) -> Optional[dict]:
    """Attributes for the target dataset's ledger copy of `row`.

    A full re-run would serialize `belongs_to_set` (and the document's
    `source_node_set`) from the data item's *current* node_set, so the copy
    rewrites those fields to `tags` instead of inheriting the source
    dataset's values. Rows the pipeline never tags are copied as-is.
    """
    if not isinstance(row.attributes, dict):
        return row.attributes
    if not isinstance(row.attributes.get("belongs_to_set"), list):
        return row.attributes

    attributes = copy.deepcopy(row.attributes)
    attributes["belongs_to_set"] = list(tags) if tags else None
    if "source_node_set" in attributes:
        attributes["source_node_set"] = ", ".join(tags) if tags else None
    return attributes


def _build_node_set_artifacts(
    tags: List[str], taggable_slugs: List[str]
) -> Tuple[List[NodeSet], List[Tuple[UUID, UUID, str, Dict[str, Any]]]]:
    """The NodeSet anchor points and `belongs_to_set` edges a full run would
    create for the target's tags: one edge from every tagged node to every
    target NodeSet anchor."""
    node_set_points = [
        NodeSet(id=generate_node_id(f"NodeSet:{tag_name}"), name=tag_name) for tag_name in tags
    ]

    now_text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    belongs_to_set_edges: List[Tuple[UUID, UUID, str, Dict[str, Any]]] = [
        (
            UUID(slug),
            node_set_point.id,
            "belongs_to_set",
            {
                "source_node_id": slug,
                "target_node_id": str(node_set_point.id),
                "relationship_name": "belongs_to_set",
                "updated_at": now_text,
            },
        )
        for slug in taggable_slugs
        for node_set_point in node_set_points
    ]
    return node_set_points, belongs_to_set_edges


async def _apply_node_set_membership(
    graph_engine,
    vector_engine,
    tags: List[str],
    taggable_slugs: List[str],
    node_set_points: List[NodeSet],
    belongs_to_set_edges: List[Tuple[UUID, UUID, str, Dict[str, Any]]],
    source_ref_key: Optional[str] = None,
    pipeline_run_id: Optional[str] = None,
) -> None:
    """Apply the target dataset's NodeSet membership to graph + vector stores.

    When `source_ref_key` is provided (graph-provenance mode) the newly
    created NodeSet anchors and `belongs_to_set` edges are stamped with it,
    the same way add_data_points folds provenance into pipeline writes.
    """
    if not tags or not taggable_slugs:
        return None

    await vector_engine.add_belongs_to_set_tags(tags, taggable_slugs)

    await graph_engine.add_nodes(
        node_set_points, source_ref_key=source_ref_key, pipeline_run_id=pipeline_run_id
    )
    await graph_engine.add_belongs_to_set_tags(tags, taggable_slugs)
    await graph_engine.add_edges(
        [
            (str(source_id), str(target_id), relationship_name, properties)
            for source_id, target_id, relationship_name, properties in belongs_to_set_edges
        ],
        source_ref_key=source_ref_key,
        pipeline_run_id=pipeline_run_id,
    )
    return None


async def _link_via_relational_ledger(
    graph_engine,
    vector_engine,
    db_engine,
    node_rows: List[Node],
    edge_rows: List[Edge],
    data: Data,
    target_dataset: Dataset,
    user: User,
    tags: List[str],
    pipeline_run_id: Optional[UUID],
) -> bool:
    """Link on relational-ledger provenance: tag stores, copy ledger rows."""
    # The pipeline only tags DataPoints that carry a belongs_to_set list
    # (documents, chunks, summaries); the ledger attributes record which
    # ones those were, so linking tags exactly the same nodes.
    taggable_slugs = [
        str(row.slug)
        for row in node_rows
        if row.type != "NodeSet"
        and isinstance(row.attributes, dict)
        and isinstance(row.attributes.get("belongs_to_set"), list)
    ]

    node_set_points, belongs_to_set_edges = _build_node_set_artifacts(tags, taggable_slugs)

    await _apply_node_set_membership(
        graph_engine, vector_engine, tags, taggable_slugs, node_set_points, belongs_to_set_edges
    )

    # ── Ledger: record the (target_dataset, data) provenance rows ──────
    async with db_engine.get_async_session() as session:
        copied_node_rows = [
            {
                "id": uuid5(
                    NAMESPACE_OID,
                    str(user.tenant_id)
                    + str(user.id)
                    + str(target_dataset.id)
                    + str(data.id)
                    + str(row.slug),
                ),
                "slug": row.slug,
                "user_id": user.id,
                "data_id": data.id,
                "dataset_id": target_dataset.id,
                "pipeline_run_id": pipeline_run_id,
                "type": row.type,
                "indexed_fields": row.indexed_fields,
                "label": row.label,
                "attributes": _copied_node_attributes(row, tags),
            }
            for row in node_rows
            # A full run would only create NodeSet rows for the target's
            # own tags — those are written below via upsert_nodes.
            if row.type != "NodeSet"
        ]
        for start_index in range(0, len(copied_node_rows), LEDGER_BATCH_SIZE):
            batch = copied_node_rows[start_index : start_index + LEDGER_BATCH_SIZE]
            await session.execute(
                insert(Node).values(batch).on_conflict_do_nothing(index_elements=["id"])
            )

        copied_edge_rows = [
            {
                "id": uuid5(
                    NAMESPACE_OID,
                    str(user.tenant_id)
                    + str(user.id)
                    + str(target_dataset.id)
                    + str(row.source_node_id)
                    + str(row.relationship_name)
                    + str(row.destination_node_id),
                ),
                "slug": row.slug,
                "user_id": user.id,
                "data_id": data.id,
                "dataset_id": target_dataset.id,
                "pipeline_run_id": pipeline_run_id,
                "source_node_id": row.source_node_id,
                "destination_node_id": row.destination_node_id,
                "relationship_name": row.relationship_name,
                "label": row.label,
                "attributes": row.attributes,
            }
            for row in edge_rows
            # Source belongs_to_set edges point at the source dataset's
            # NodeSets; the target's are created below via upsert_edges.
            if row.relationship_name != "belongs_to_set"
        ]
        for start_index in range(0, len(copied_edge_rows), LEDGER_BATCH_SIZE):
            batch = copied_edge_rows[start_index : start_index + LEDGER_BATCH_SIZE]
            await session.execute(
                insert(Edge).values(batch).on_conflict_do_nothing(index_elements=["id"])
            )

        if tags and taggable_slugs:
            await upsert_nodes(
                node_set_points,
                tenant_id=user.tenant_id,
                user_id=user.id,
                dataset_id=target_dataset.id,
                data_id=data.id,
                session=session,
                pipeline_run_id=pipeline_run_id,
            )
            await upsert_edges(
                belongs_to_set_edges,
                tenant_id=user.tenant_id,
                user_id=user.id,
                dataset_id=target_dataset.id,
                data_id=data.id,
                session=session,
                pipeline_run_id=pipeline_run_id,
            )

        await session.commit()

    logger.info(
        "Linked existing extraction of data %s to dataset %s via relational ledger "
        "(%d nodes, %d edges, %d tagged with %s) — pipeline run skipped.",
        data.id,
        target_dataset.id,
        len(copied_node_rows),
        len(copied_edge_rows),
        len(taggable_slugs),
        tags,
    )
    return True


async def _link_via_graph_provenance(
    graph_engine,
    vector_engine,
    data: Data,
    source_dataset_id: UUID,
    target_dataset: Dataset,
    tags: List[str],
    pipeline_run_id: Optional[UUID],
) -> bool:
    """Link on graph-native provenance: tag stores, attach the target's
    source refs to the already-stamped nodes/edges.

    On graph-provenance graphs there is no relational ledger — ownership
    lives in each node/edge's `source_ref_keys`. A full run for the target
    dataset would stamp every artifact it writes with
    `make_source_ref_key(target_dataset, data)`; linking attaches exactly
    that ref to the artifacts already stamped for the source pair, so
    per-dataset delete and rollback keep working.
    """
    source_ref_key = make_source_ref_key(source_dataset_id, data.id)
    node_ids = await graph_engine.find_nodes_by_source_ref(source_ref_key)
    if not node_ids:
        logger.debug(
            "Cross-dataset reuse skipped: no graph-provenance artifacts for data %s in dataset %s.",
            data.id,
            source_dataset_id,
        )
        return False

    node_properties = await graph_engine.extract_nodes([str(node_id) for node_id in node_ids])

    # A full run for the target would stamp everything it writes except the
    # source dataset's own NodeSet anchor.
    linkable_nodes = [
        properties for properties in node_properties if properties.get("type") != "NodeSet"
    ]
    if not linkable_nodes:
        return False
    linkable_ids = [str(properties["id"]) for properties in linkable_nodes]

    # Same taggability rule as the ledger path: only nodes the pipeline
    # tagged (their belongs_to_set is a list) gain the target's tags.
    taggable_slugs = [
        str(properties["id"])
        for properties in linkable_nodes
        if isinstance(properties.get("belongs_to_set"), list)
    ]

    edge_identities = await graph_engine.find_edges_by_source_ref(source_ref_key)
    # Source belongs_to_set edges point at the source dataset's NodeSet
    # anchor; the target's own are created freshly below, already stamped.
    linkable_edges = [
        edge for edge in edge_identities if edge.relationship_name != "belongs_to_set"
    ]

    target_ref_key = make_source_ref_key(target_dataset.id, data.id)
    run_id_text = str(pipeline_run_id) if pipeline_run_id else None

    node_set_points, belongs_to_set_edges = _build_node_set_artifacts(tags, taggable_slugs)

    await _apply_node_set_membership(
        graph_engine,
        vector_engine,
        tags,
        taggable_slugs,
        node_set_points,
        belongs_to_set_edges,
        source_ref_key=target_ref_key,
        pipeline_run_id=run_id_text,
    )

    await graph_engine.attach_node_source_refs(linkable_ids, [target_ref_key], run_id_text)
    if linkable_edges:
        await graph_engine.attach_edge_source_refs(linkable_edges, [target_ref_key], run_id_text)

    logger.info(
        "Linked existing extraction of data %s to dataset %s via graph provenance "
        "(%d nodes, %d edges, %d tagged with %s) — pipeline run skipped.",
        data.id,
        target_dataset.id,
        len(linkable_ids),
        len(linkable_edges),
        len(taggable_slugs),
        tags,
    )
    return True


async def link_data_to_dataset(
    data: Data,
    source_dataset_id: UUID,
    target_dataset: Dataset,
    user: User,
    pipeline_run_id: Optional[UUID] = None,
) -> bool:
    """Link `data`'s already-materialized artifacts to `target_dataset`.

    When the same data item was already processed for another dataset in the
    same graph/vector database, everything the pipeline would produce for it
    already exists — nodes, edges, embeddings — because artifact ids are
    deterministic. What a run for `target_dataset` would add is only:

      1. the target dataset's NodeSet membership (`belongs_to_set` property
         on graph nodes and vector payloads, plus `belongs_to_set` edges and
         the NodeSet anchor nodes), and
      2. per-dataset provenance — relational ledger rows keyed by
         `(target_dataset, data)`, or `source_ref_keys` stamps on
         graph-provenance graphs — so deletion and rollback keep working.

    This function applies exactly those writes, letting the caller skip the
    full (LLM + embedding) pipeline. Every operation is additive and
    idempotent, so a failure mid-way is safe: the caller falls back to full
    processing, which converges the stores to the same state.

    Both provenance modes are supported: graphs with relational ledger rows
    copy them re-keyed to the target, and graph-provenance graphs (marked
    via graph metadata) attach the target's source refs instead.

    Returns True when the artifacts were linked; False when linking is not
    supported for the current configuration (caller must fall back):
      - per-dataset databases (`ENABLE_BACKEND_ACCESS_CONTROL`) — the
        artifacts don't exist in the target dataset's database,
      - no provenance for the source pair in either mode (e.g. legacy data),
      - graph/vector adapter without `add_belongs_to_set_tags` support.
    """
    if backend_access_control_enabled():
        logger.debug(
            "Cross-dataset reuse skipped: per-dataset databases are enabled "
            "(ENABLE_BACKEND_ACCESS_CONTROL)."
        )
        return False

    try:
        graph_engine = await get_graph_engine()
        vector_engine = await get_vector_engine_async()

        if not _supports_tag_add(graph_engine, vector_engine):
            logger.debug(
                "Cross-dataset reuse skipped: the configured graph or vector "
                "adapter does not implement add_belongs_to_set_tags."
            )
            return False

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            node_rows = (
                await session.scalars(
                    select(Node).where(
                        Node.data_id == data.id, Node.dataset_id == source_dataset_id
                    )
                )
            ).all()
            edge_rows = (
                await session.scalars(
                    select(Edge).where(
                        Edge.data_id == data.id, Edge.dataset_id == source_dataset_id
                    )
                )
            ).all()

        tags = _data_node_set(data)

        if node_rows:
            return await _link_via_relational_ledger(
                graph_engine,
                vector_engine,
                db_engine,
                node_rows,
                edge_rows,
                data,
                target_dataset,
                user,
                tags,
                pipeline_run_id,
            )

        if await stores_provenance_in_graph(graph_engine):
            return await _link_via_graph_provenance(
                graph_engine,
                vector_engine,
                data,
                source_dataset_id,
                target_dataset,
                tags,
                pipeline_run_id,
            )

        logger.debug(
            "Cross-dataset reuse skipped: no provenance ledger rows for "
            "data %s in dataset %s and the graph does not store provenance "
            "(legacy data).",
            data.id,
            source_dataset_id,
        )
        return False

    except (NotImplementedError, UnsupportedProvenanceCapability):
        logger.debug(
            "Cross-dataset reuse skipped: adapter does not support the required "
            "tagging/provenance operations."
        )
        return False
    except Exception as error:
        # All link writes are additive and idempotent; falling back to full
        # processing converges the stores regardless of where this failed.
        logger.warning(
            "Cross-dataset reuse failed for data %s into dataset %s, "
            "falling back to full processing: %s",
            data.id,
            target_dataset.id,
            error,
        )
        return False
