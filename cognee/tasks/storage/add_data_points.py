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
    deduplicate_nodes_and_edges,
    ensure_default_edge_properties,
    get_graph_from_model,
)
from .index_data_points import index_data_points
from .chunk_ownership import collect_chunk_ownership
from .index_graph_edges import index_graph_edges
from cognee.modules.engine.models import Triplet
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.exceptions import (
    InvalidDataPointsInAddDataPointsError,
)
from cognee.modules.provenance.edge_evidence.capture import capture_graph_provenance
from ...modules.engine.utils import generate_node_id

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("add_data_points")


def _group_by_all_keys(owner_map: dict) -> dict:
    """Artifacts that need the same FULL ref key set, grouped into one call.

    For artifacts this batch does not write (relationship edges the graph
    already held), nothing folds: every owner must be attached.
    """
    groups: dict = {}
    for artifact, owners in owner_map.items():
        if owners:
            groups.setdefault(tuple(owners), []).append(artifact)
    return groups


def _group_by_extra_keys(owner_map: dict) -> dict:
    """Artifacts that need the same extra ref keys, grouped into one call.

    An artifact's FIRST owner folds into the statement that writes the row, so
    only ``owners[1:]`` need a separate attach. Returns
    ``{(key, ...): [artifact, ...]}``.
    """
    groups: dict = {}
    for artifact, owners in owner_map.items():
        extra = tuple(owners[1:])
        if extra:
            groups.setdefault(extra, []).append(artifact)
    return groups


@task_summary("Stored {n} data point(s)")
async def add_data_points(
    data_points: List[DataPoint],
    custom_edges: Optional[List] = None,
    embed_triplets: bool = False,
    ctx: Optional["PipelineContext"] = None,
    graph_only: bool = False,
) -> List[DataPoint]:
    """
    Add a batch of data points to the graph database by extracting nodes and edges,
    deduplicating them, and indexing them for retrieval.

    Args:
        data_points: Data points to process and insert into the graph.
        custom_edges: Custom edges between datapoints.
        embed_triplets: If True, creates and indexes triplet embeddings.
        ctx: Pipeline runtime context (user, dataset, data_item).
        graph_only: Persist graph nodes and edges without initializing or writing
            a vector engine. Intended for deterministic extraction pipelines.
    """
    user = ctx.user if ctx else None
    data_item = ctx.data_item if ctx else None
    dataset = ctx.dataset if ctx else None
    pipeline_run_id = ctx.pipeline_run_id if ctx else None

    if not isinstance(data_points, list):
        raise InvalidDataPointsInAddDataPointsError("data_points must be a list.")
    if not all(isinstance(dp, DataPoint) for dp in data_points):
        raise InvalidDataPointsInAddDataPointsError("data_points: each item must be a DataPoint.")
    if graph_only and embed_triplets:
        raise InvalidDataPointsInAddDataPointsError(
            "embed_triplets cannot be enabled when graph_only is True."
        )

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

    if graph_only:
        from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

        graph_engine = await get_graph_engine()
        vector_engine = None
        use_hybrid = False
    else:
        unified = await get_unified_engine()
        graph_engine = unified.graph
        vector_engine = unified.vector
        use_hybrid = unified.has_capability(EngineCapability.HYBRID_WRITE)

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
    #
    # CHUNK-SCOPED OWNERSHIP (source_ref:v2): when the batch contains
    # DocumentChunks, every node/edge produced by a chunk is stamped with that
    # chunk's v2 ref — written in its FIRST owner's batch (fold stays atomic),
    # remaining owners attached after. Output no chunk produced (the document
    # node, NodeSet tags) keeps the document-scoped v1 ref. Deletion can then
    # operate at chunk scope while shared output survives to its last owner.
    fold_source_ref_key = None
    fold_run_arg = None
    ownership = None
    if stores_provenance:
        fold_source_ref_key = make_source_ref_key(dataset.id, data_id)
        fold_run_arg = str(pipeline_run_id) if pipeline_run_id else None
        ownership = await collect_chunk_ownership(data_points, dataset.id, data_id)
        if not ownership.has_chunks:
            ownership = None

    def _group_nodes():
        groups: dict = {}
        for node in nodes:
            owners = ownership.node_owners.get(str(node.id)) if ownership else None
            key = owners[0] if owners else fold_source_ref_key
            groups.setdefault(key, []).append(node)
        return groups

    def _group_edges(edge_list):
        groups: dict = {}
        for edge in edge_list:
            owners = (
                ownership.edge_owners.get((str(edge[0]), str(edge[1]), str(edge[2])))
                if ownership
                else None
            )
            key = owners[0] if owners else fold_source_ref_key
            groups.setdefault(key, []).append(edge)
        return groups

    # Backends with per-row ref support take the whole batch in ONE statement,
    # each row carrying its own (first-owner) key — measured 8x faster on the
    # default backend than one add_nodes/add_edges call per owner group.
    per_row_refs = bool(ownership and getattr(graph_engine, "supports_per_row_source_refs", False))

    async def _write_nodes_grouped():
        if per_row_refs:
            mapping = {
                str(node.id): (ownership.node_owners.get(str(node.id)) or [fold_source_ref_key])[0]
                for node in nodes
            }
            await graph_engine.add_nodes(
                nodes, source_ref_key=mapping, pipeline_run_id=fold_run_arg
            )
        else:
            for key, group in _group_nodes().items():
                await graph_engine.add_nodes(
                    group, source_ref_key=key, pipeline_run_id=fold_run_arg
                )
        if ownership:
            # Group by the SET of extra keys, not by individual key. Each call
            # is a lock-serialized read-then-write query pair, so grouping per
            # key costs one round-trip per owning chunk — linear in chunk count
            # (measured: 39 node calls at 40 chunks, 99 at 100). Artifacts
            # needing the same keys collapse into one call instead, making the
            # cost scale with distinct sharing patterns rather than document
            # length. Every artifact still receives exactly its own key set.
            for keys, node_ids in _group_by_extra_keys(ownership.node_owners).items():
                await graph_engine.attach_node_source_refs(node_ids, list(keys), fold_run_arg)

    async def _write_edges_grouped(edge_list):
        if per_row_refs:
            mapping = {
                (str(e[0]), str(e[1]), str(e[2])): (
                    ownership.edge_owners.get((str(e[0]), str(e[1]), str(e[2])))
                    or [fold_source_ref_key]
                )[0]
                for e in edge_list
            }
            await graph_engine.add_edges(
                edge_list, source_ref_key=mapping, pipeline_run_id=fold_run_arg
            )
        else:
            for key, group in _group_edges(edge_list).items():
                await graph_engine.add_edges(
                    group, source_ref_key=key, pipeline_run_id=fold_run_arg
                )
        if ownership:
            batch_keys = {(str(e[0]), str(e[1]), str(e[2])) for e in edge_list}
            in_batch = {
                edge_key: owners
                for edge_key, owners in ownership.edge_owners.items()
                if edge_key in batch_keys
            }
            # Grouped by key set, like the nodes above.
            for keys, edge_keys in _group_by_extra_keys(in_batch).items():
                identities = [EdgeIdentity(key[0], key[1], key[2]) for key in edge_keys]
                await graph_engine.attach_edge_source_refs(identities, list(keys), fold_run_arg)
            # Relationship edges this batch's chunks produced but did not write
            # because the graph already held them. They gain their new owners
            # by ref attach alone — no rewrite, so stored edge properties
            # (weights, feedback) stay untouched — otherwise the edge would be
            # deleted with its FIRST producer while these chunks still state it.
            existing = {
                edge_key: owners
                for edge_key, owners in ownership.edge_owners.items()
                if edge_key not in batch_keys
            }
            for keys, edge_keys in _group_by_all_keys(existing).items():
                identities = [EdgeIdentity(key[0], key[1], key[2]) for key in edge_keys]
                await graph_engine.attach_edge_source_refs(identities, list(keys), fold_run_arg)

    # GRAPH BEFORE VECTORS, never concurrently. The two stores fail
    # independently, and only one of the two orders is recoverable:
    #
    #   graph ok, vector fails -> the artifact carries its source ref, so
    #     rollback and delete can both find it. Self-healing.
    #   vector ok, graph fails -> the point exists with NO ref anywhere in the
    #     graph. Nothing can discover it again, while CHUNKS retrieval reads
    #     the vector collection directly and happily returns it — content from
    #     a run that reported failure, served to users, permanently.
    #
    # Concurrency bought a little latency and made the second case reachable
    # on every write, so the writes are sequenced instead.
    if use_hybrid:
        await graph_engine.add_nodes_with_vectors(nodes)
    elif graph_only:
        await _write_nodes_grouped()
    else:
        await _write_nodes_grouped()
        await index_data_points(
            [node.model_copy(deep=True) for node in nodes],
            vector_engine=vector_engine,
        )

    if use_hybrid:
        await graph_engine.add_edges_with_vectors(edges)
    elif graph_only:
        await _write_edges_grouped(edges)
    else:
        await _write_edges_grouped(edges)
        await index_graph_edges(edges, vector_engine=vector_engine)

    if custom_edges:
        # This must be handled separately from datapoint edges, created a task in linear to dig deeper but (COG-3488)
        # Note: custom_edges is already normalized (with nodes) above, before the
        # rollback-ledger upsert, so no second ensure_default_edge_properties here.
        if use_hybrid:
            await graph_engine.add_edges_with_vectors(custom_edges)
        elif graph_only:
            await graph_engine.add_edges(
                custom_edges,
                source_ref_key=fold_source_ref_key,
                pipeline_run_id=fold_run_arg,
            )
        else:
            await graph_engine.add_edges(
                custom_edges,
                source_ref_key=fold_source_ref_key,
                pipeline_run_id=fold_run_arg,
            )
            await index_graph_edges(custom_edges, vector_engine=vector_engine)

        edges.extend(custom_edges)

    if stores_provenance and use_hybrid:
        # Hybrid backends write nodes/edges and their vectors in one call that
        # cannot yet fold provenance, so stamp the source refs in a separate
        # attach pass (chunk-scoped where ownership exists, document-scoped
        # otherwise). This keeps a write-then-attach window for hybrid graphs
        # only; the non-hybrid path above is already atomic.
        run_arg = str(pipeline_run_id) if pipeline_run_id else None

        attach_nodes: dict = {}
        for node in nodes:
            owners = ownership.node_owners.get(str(node.id)) if ownership else None
            for key in owners or [fold_source_ref_key]:
                attach_nodes.setdefault(key, []).append(str(node.id))
        for ref_key, node_ids in attach_nodes.items():
            await graph_engine.attach_node_source_refs(node_ids, [ref_key], run_arg)

        attach_edges: dict = {}
        written_edge_keys = set()
        for edge in edges:
            edge_key = (str(edge[0]), str(edge[1]), str(edge[2]))
            written_edge_keys.add(edge_key)
            owners = ownership.edge_owners.get(edge_key) if ownership else None
            for key in owners or [fold_source_ref_key]:
                attach_edges.setdefault(key, []).append(
                    EdgeIdentity(edge_key[0], edge_key[1], edge_key[2])
                )
        # Produced-but-existing relationship edges (see the non-hybrid path).
        if ownership:
            for edge_key, owners in ownership.edge_owners.items():
                if edge_key in written_edge_keys:
                    continue
                for key in owners:
                    attach_edges.setdefault(key, []).append(
                        EdgeIdentity(edge_key[0], edge_key[1], edge_key[2])
                    )
        for ref_key, identities in attach_edges.items():
            await graph_engine.attach_edge_source_refs(identities, [ref_key], run_arg)

    if embed_triplets:
        triplets = _create_triplets_from_graph(nodes, edges)
        if triplets:
            await index_data_points(triplets, vector_engine=vector_engine)
            logger.info(f"Created and indexed {len(triplets)} triplets from graph structure")

    # Capture only after graph/vector writes succeeded. This is memory-only for
    # normal documents and is flushed once at data-item completion; very large
    # documents use a bounded bulk flush configured by EDGE_EVIDENCE_FLUSH_THRESHOLD.
    # The original ``data_points`` are passed, not the expanded ``nodes``: expansion
    # rebuilds nodes as stripped copies, and the chunks a cognify batch carries are
    # nested under its TextSummary objects — capture walks the object graph itself.
    await capture_graph_provenance(data_points, edges, ctx)

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
