import asyncio
from typing import TYPE_CHECKING, Dict, List, Optional

from cognee.modules.pipelines.tasks.task import task_summary
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    make_source_ref_key,
)
from cognee.infrastructure.databases.provenance.markers import (
    ensure_graph_native_for_new_graph,
)
from cognee.infrastructure.databases.relational import get_async_session
from cognee.modules.graph.methods import upsert_edges, upsert_nodes
from cognee.modules.graph.utils import (
    deduplicate_nodes_and_edges,
    ensure_default_edge_properties,
    get_graph_from_model,
)
from .index_data_points import index_data_points
from .index_graph_edges import index_graph_edges
from cognee.modules.engine.models import Triplet
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.exceptions import (
    InvalidDataPointsInAddDataPointsError,
)
from ...modules.engine.utils import generate_node_id

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("add_data_points")


@task_summary("Stored {n} data point(s)")
async def add_data_points(
    data_points: List[DataPoint],
    custom_edges: Optional[List] = None,
    embed_triplets: bool = False,
    ctx: Optional["PipelineContext"] = None,
) -> List[DataPoint]:
    """
    Add a batch of data points to the graph database by extracting nodes and edges,
    deduplicating them, and indexing them for retrieval.

    Args:
        data_points: Data points to process and insert into the graph.
        custom_edges: Custom edges between datapoints.
        embed_triplets: If True, creates and indexes triplet embeddings.
        ctx: Pipeline runtime context (user, dataset, data_item).
    """
    user = ctx.user if ctx else None
    data_item = ctx.data_item if ctx else None
    dataset = ctx.dataset if ctx else None
    pipeline_run_id = ctx.pipeline_run_id if ctx else None

    if not isinstance(data_points, list):
        raise InvalidDataPointsInAddDataPointsError("data_points must be a list.")
    if not all(isinstance(dp, DataPoint) for dp in data_points):
        raise InvalidDataPointsInAddDataPointsError("data_points: each item must be a DataPoint.")

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    results = await asyncio.gather(
        *[
            get_graph_from_model(
                data_point,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for data_point in data_points
        ]
    )

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    nodes, edges = deduplicate_nodes_and_edges(nodes, edges)

    edges = ensure_default_edge_properties(edges, nodes=nodes)
    custom_edges = (
        ensure_default_edge_properties(custom_edges, nodes=nodes)
        if isinstance(custom_edges, list) and custom_edges
        else None
    )

    unified = await get_unified_engine()
    graph_engine = unified.graph
    vector_engine = unified.vector
    use_hybrid = unified.has_capability(EngineCapability.HYBRID_WRITE)

    is_graph_native = False
    if user and dataset and data_item:
        # Graph-native graphs (empty graphs marked via graph metadata) carry
        # their provenance in the graph itself, so they skip the relational
        # rollback ledger entirely. On backends that implement provenance
        # (e.g. Ladybug + LanceDB) a fresh empty graph IS marked here and takes
        # the graph-native path; backends without provenance support raise on
        # set_graph_metadata, so this stays False and the ledger path runs.
        #
        # NOTE: the graph/vector writes below happen before the source refs are
        # attached (see end of this function). If a write succeeds but the attach
        # fails, the artifacts have neither ledger rows nor graph provenance and
        # cannot be found by delete/rollback. True atomicity needs adapter-level
        # transactional attach (a Part 1 follow-up); until then the attach raising
        # marks the run failed.
        is_graph_native = await ensure_graph_native_for_new_graph(graph_engine)

        if not is_graph_native:
            # Single session for all upserts: one transaction, one commit. The
            # rollback ledger is written BEFORE the graph/vector writes so a
            # failed write can always be swept by the rollback handler.
            async with get_async_session() as session:
                await upsert_nodes(
                    nodes,
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    dataset_id=dataset.id,
                    data_id=data_item.id,
                    session=session,
                    pipeline_run_id=pipeline_run_id,
                )
                await upsert_edges(
                    edges,
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    dataset_id=dataset.id,
                    data_id=data_item.id,
                    session=session,
                    pipeline_run_id=pipeline_run_id,
                )
                if custom_edges:
                    await upsert_edges(
                        custom_edges,
                        tenant_id=user.tenant_id,
                        user_id=user.id,
                        dataset_id=dataset.id,
                        data_id=data_item.id,
                        session=session,
                        pipeline_run_id=pipeline_run_id,
                    )
                await session.commit()

    if use_hybrid:
        await graph_engine.add_nodes_with_vectors(nodes)
    else:
        await asyncio.gather(
            graph_engine.add_nodes(nodes),
            index_data_points(
                [node.model_copy(deep=True) for node in nodes],
                vector_engine=vector_engine,
            ),
        )

    if use_hybrid:
        await graph_engine.add_edges_with_vectors(edges)
    else:
        await asyncio.gather(
            graph_engine.add_edges(edges), index_graph_edges(edges, vector_engine=vector_engine)
        )

    if custom_edges:
        # This must be handled separately from datapoint edges, created a task in linear to dig deeper but (COG-3488)
        # Note: custom_edges is already normalized (with nodes) above, before the
        # rollback-ledger upsert, so no second ensure_default_edge_properties here.
        if use_hybrid:
            await graph_engine.add_edges_with_vectors(custom_edges)
        else:
            await asyncio.gather(
                graph_engine.add_edges(custom_edges),
                index_graph_edges(custom_edges, vector_engine=vector_engine),
            )

        edges.extend(custom_edges)

    if is_graph_native and user and dataset and data_item:
        # Graph-native provenance: now that the graph (and vector) writes have
        # succeeded, stamp the source refs onto the written nodes/edges. A
        # single attach call per artifact kind — the adapter derives the
        # dataset_ids / run_ids / run_refs from the source ref key + run id.
        source_ref_key = make_source_ref_key(dataset.id, data_item.id)
        run_arg = str(pipeline_run_id) if pipeline_run_id else None

        node_ids = [str(node.id) for node in nodes]
        edge_ids = [EdgeIdentity(str(edge[0]), str(edge[1]), edge[2]) for edge in edges]

        if node_ids:
            await graph_engine.attach_node_source_refs(node_ids, [source_ref_key], run_arg)
        if edge_ids:
            await graph_engine.attach_edge_source_refs(edge_ids, [source_ref_key], run_arg)

    if embed_triplets:
        triplets = _create_triplets_from_graph(nodes, edges)
        if triplets:
            await index_data_points(triplets, vector_engine=vector_engine)
            logger.info(f"Created and indexed {len(triplets)} triplets from graph structure")

    return data_points


def _extract_embeddable_text_from_datapoint(data_point: DataPoint) -> str:
    """
    Extract embeddable text from a DataPoint using its index_fields metadata.
    Uses the same approach as index_data_points.

    Parameters:
    -----------
        - data_point (DataPoint): The data point to extract text from.

    Returns:
    --------
        - str: Concatenated string of all embeddable property values, or empty string if none found.
    """
    if not data_point or not hasattr(data_point, "metadata"):
        return ""

    index_fields = data_point.metadata.get("index_fields", [])
    if not index_fields:
        return ""

    embeddable_values = []
    for field_name in index_fields:
        field_value = getattr(data_point, field_name, None)
        if field_value is not None:
            field_value = str(field_value).strip()

            if field_value:
                embeddable_values.append(field_value)

    return " ".join(embeddable_values) if embeddable_values else ""


def _create_triplets_from_graph(nodes: List[DataPoint], edges: List[tuple]) -> List[Triplet]:
    """
    Create Triplet objects from graph nodes and edges.

    This function processes graph edges and their corresponding nodes to create
    triplet datapoints with embeddable text, similar to the triplet embeddings pipeline.

    Parameters:
    -----------
        - nodes (List[DataPoint]): List of graph nodes extracted from data points
        - edges (List[tuple]): List of edge tuples in format
          (source_node_id, target_node_id, relationship_name, properties_dict)
          Note: All edges including those from DocumentChunk.contains are already extracted
          by get_graph_from_model and included in this list.

    Returns:
    --------
        - List[Triplet]: List of Triplet objects ready for indexing
    """
    node_map: Dict[str, DataPoint] = {}
    for node in nodes:
        if hasattr(node, "id"):
            node_id = str(node.id)
            if node_id not in node_map:
                node_map[node_id] = node

    triplets = []
    skipped_count = 0
    seen_ids = set()

    for edge_tuple in edges:
        if len(edge_tuple) < 4:
            continue

        source_node_id, target_node_id, relationship_name, edge_properties = (
            edge_tuple[0],
            edge_tuple[1],
            edge_tuple[2],
            edge_tuple[3],
        )

        source_node = node_map.get(str(source_node_id))
        target_node = node_map.get(str(target_node_id))

        if not source_node or not target_node or relationship_name is None:
            skipped_count += 1
            continue

        source_node_text = _extract_embeddable_text_from_datapoint(source_node)
        target_node_text = _extract_embeddable_text_from_datapoint(target_node)

        relationship_text = ""
        if isinstance(edge_properties, dict):
            edge_text = edge_properties.get("edge_text")
            if edge_text and isinstance(edge_text, str) and edge_text.strip():
                relationship_text = edge_text.strip()

        if not relationship_text and relationship_name:
            relationship_text = relationship_name

        if not source_node_text and not relationship_text and not relationship_name:
            skipped_count += 1
            continue

        embeddable_text = f"{source_node_text} -› {relationship_text}-›{target_node_text}".strip()

        triplet_id = generate_node_id(str(source_node_id) + relationship_name + str(target_node_id))

        if triplet_id in seen_ids:
            continue
        seen_ids.add(triplet_id)

        triplets.append(
            Triplet(
                id=triplet_id,
                from_node_id=str(source_node_id),
                to_node_id=str(target_node_id),
                text=embeddable_text,
            )
        )

    return triplets
