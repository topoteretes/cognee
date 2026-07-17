import asyncio
from typing import TYPE_CHECKING, Dict, List, Optional

from cognee.modules.pipelines.tasks.task import task_summary
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    data_item_id,
    make_source_ref_key,
)
from cognee.infrastructure.databases.provenance.markers import (
    mark_graph_provenance_if_empty,
)
from cognee.infrastructure.databases.relational import get_async_session
from cognee.modules.graph.methods import upsert_edges, upsert_nodes
from cognee.modules.graph.utils import (
    FUZZY_DEDUP_THRESHOLD,
    deduplicate_nodes_and_edges,
    ensure_default_edge_properties,
    get_graph_from_model,
    resolve_fuzzy_duplicate_entities,
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
    fuzzy_entity_dedup: bool = False,
    fuzzy_entity_dedup_threshold: float = FUZZY_DEDUP_THRESHOLD,
) -> List[DataPoint]:
    """
    Add a batch of data points to the graph database by extracting nodes and edges,
    deduplicating them, and indexing them for retrieval.

    Args:
        data_points: Data points to process and insert into the graph.
        custom_edges: Custom edges between datapoints.
        embed_triplets: If True, creates and indexes triplet embeddings.
        ctx: Pipeline runtime context (user, dataset, data_item).
        fuzzy_entity_dedup: When True, Entity nodes whose names embed close
            together in this batch (e.g. "OpenAI" / "OpenAI Inc.") are linked
            with a ``merged_into`` edge (see resolve_fuzzy_duplicate_entities).
            Off by default; most batches have no fuzzy duplicates.
        fuzzy_entity_dedup_threshold: Minimum cosine similarity for a fuzzy
            merge. Only used when ``fuzzy_entity_dedup`` is True.
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

    unified = await get_unified_engine()
    graph_engine = unified.graph
    vector_engine = unified.vector
    use_hybrid = unified.has_capability(EngineCapability.HYBRID_WRITE)

    nodes, edges = deduplicate_nodes_and_edges(nodes, edges)

    # Approach B — opt-in embedding-similarity fuzzy dedup (issue #3628). When
    # enabled, Entity nodes whose names embed close together in this batch (e.g.
    # "OpenAI" / "OpenAI Inc.") are linked with a `merged_into` edge, so the
    # duplication is recorded in the graph — non-destructively and reversibly.
    # Off by default; most batches have no fuzzy duplicates and pay nothing.
    if fuzzy_entity_dedup:
        merge_edges = await resolve_fuzzy_duplicate_entities(
            nodes, vector_engine, similarity_threshold=fuzzy_entity_dedup_threshold
        )
        if merge_edges:
            edges = edges + merge_edges
            logger.info("Fuzzy dedup: linked %d duplicate entity pair(s)", len(merge_edges))

    edges = ensure_default_edge_properties(edges, nodes=nodes)
    custom_edges = (
        ensure_default_edge_properties(custom_edges, nodes=nodes)
        if isinstance(custom_edges, list) and custom_edges
        else None
    )

    # Provenance needs a concrete (dataset, data) pair. data_item_id resolves
    # the id whether data_item is a relational Data (.id) or an ingestion
    # DataItem (.data_id); it is None for items that carry neither (a raw
    # file/text item, or the CogneeGraph memify passes in), in which case there
    # is nothing to attribute and both the ledger and graph-fold paths are skipped.
    data_id = data_item_id(data_item)
    stores_provenance = False
    if user and dataset and data_id is not None:
        # Graph-provenance graphs (empty graphs marked via graph metadata) carry
        # their provenance in the graph itself, so they skip the relational
        # rollback ledger entirely. On backends that implement provenance
        # (e.g. Ladybug + LanceDB) a fresh empty graph IS marked here and takes
        # the graph-provenance path; backends without provenance support raise on
        # set_graph_metadata, so this stays False and the ledger path runs.
        #
        # On the non-hybrid path the provenance source refs are folded into the
        # graph write below (atomic — no window where an artifact exists without
        # its provenance). Hybrid backends still attach in a second pass and keep
        # that window; if the attach raises, the run is marked failed.
        #
        # Under COGNEE_DISTRIBUTED the graph write is diverted to a queue; the
        # provenance stamp (source_ref_key / pipeline_run_id) rides along in the
        # queue payload and is folded per data item by the graph_saving_worker,
        # so graph-native provenance works in distributed mode too.
        stores_provenance = await mark_graph_provenance_if_empty(graph_engine)

        if not stores_provenance:
            # Single session for all upserts: one transaction, one commit. The
            # rollback ledger is written BEFORE the graph/vector writes so a
            # failed write can always be swept by the rollback handler.
            async with get_async_session() as session:
                await upsert_nodes(
                    nodes,
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    dataset_id=dataset.id,
                    data_id=data_id,
                    session=session,
                    pipeline_run_id=pipeline_run_id,
                )
                await upsert_edges(
                    edges,
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    dataset_id=dataset.id,
                    data_id=data_id,
                    session=session,
                    pipeline_run_id=pipeline_run_id,
                )
                if custom_edges:
                    await upsert_edges(
                        custom_edges,
                        tenant_id=user.tenant_id,
                        user_id=user.id,
                        dataset_id=dataset.id,
                        data_id=data_id,
                        session=session,
                        pipeline_run_id=pipeline_run_id,
                    )
                await session.commit()

    # Graph provenance is folded INTO the graph write so a node/edge is
    # created and stamped in one atomic statement (no write-then-attach window,
    # no concurrent lost update — COG-5522 #4/#8). Only the non-hybrid path can
    # fold today; hybrid backends still stamp via a separate attach pass below.
    # source_ref_key stays None for non-graph-provenance writes (no provenance).
    fold_source_ref_key = None
    fold_run_arg = None
    if stores_provenance and not use_hybrid:
        fold_source_ref_key = make_source_ref_key(dataset.id, data_id)
        fold_run_arg = str(pipeline_run_id) if pipeline_run_id else None

    if use_hybrid:
        await graph_engine.add_nodes_with_vectors(nodes)
    else:
        await asyncio.gather(
            graph_engine.add_nodes(
                nodes, source_ref_key=fold_source_ref_key, pipeline_run_id=fold_run_arg
            ),
            index_data_points(
                [node.model_copy(deep=True) for node in nodes],
                vector_engine=vector_engine,
            ),
        )

    if use_hybrid:
        await graph_engine.add_edges_with_vectors(edges)
    else:
        await asyncio.gather(
            graph_engine.add_edges(
                edges, source_ref_key=fold_source_ref_key, pipeline_run_id=fold_run_arg
            ),
            index_graph_edges(edges, vector_engine=vector_engine),
        )

    if custom_edges:
        # This must be handled separately from datapoint edges, created a task in linear to dig deeper but (COG-3488)
        # Note: custom_edges is already normalized (with nodes) above, before the
        # rollback-ledger upsert, so no second ensure_default_edge_properties here.
        if use_hybrid:
            await graph_engine.add_edges_with_vectors(custom_edges)
        else:
            await asyncio.gather(
                graph_engine.add_edges(
                    custom_edges,
                    source_ref_key=fold_source_ref_key,
                    pipeline_run_id=fold_run_arg,
                ),
                index_graph_edges(custom_edges, vector_engine=vector_engine),
            )

        edges.extend(custom_edges)

    if stores_provenance and use_hybrid:
        # Hybrid backends write nodes/edges and their vectors in one call that
        # cannot yet fold provenance, so stamp the source refs in a separate
        # attach pass. This keeps a write-then-attach window for hybrid graphs
        # only; the non-hybrid path above is already atomic.
        source_ref_key = make_source_ref_key(dataset.id, data_id)
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
