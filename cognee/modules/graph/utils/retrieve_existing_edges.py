from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    graph_provenance_write_kwargs,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import _create_edge_key
from cognee.shared.data_models import KnowledgeGraph


async def retrieve_existing_edges(
    data_chunks: list[DataPoint],
    chunk_graphs: list[KnowledgeGraph],
    ctx=None,
) -> dict[str, bool]:
    """
    - LLM generated docstring
    Retrieve existing edges from the graph database to prevent duplicate edge creation.

    This function checks which edges already exist in the graph database by querying
    for various types of relationships including structural edges (exists_in, mentioned_in, is_a)
    and content-derived edges from the knowledge graphs. It returns a mapping that can be
    used to avoid creating duplicate edges during graph expansion.

    Args:
        data_chunks (list[DataPoint]): List of data point objects that serve as containers
            for the entities. Each data chunk represents a source document or data segment.
        chunk_graphs (list[KnowledgeGraph]): List of knowledge graphs corresponding to each
            data chunk. Each graph contains nodes (entities) and edges (relationships) that
            were extracted from the chunk content.
        ctx: Optional pipeline context used to attach graph-native source and run refs to
            relationships that already exist.

    Returns:
        dict[str, bool]: A mapping of existing edge keys to ``True``.

    Note:
        - The function generates several types of edges for checking:
          * Type node edges: (chunk_id, type_node_id, "exists_in")
          * Entity node edges: (chunk_id, entity_node_id, "mentioned_in")
          * Type-entity edges: (entity_node_id, type_node_id, "is_a")
          * Graph node edges: extracted from the knowledge graph relationships
        - Uses Entity.id_for() to ensure consistent node ID formatting
        - Prevents processing the same edge multiple times using a processed_edges tracker
        - The returned mapping can be used with expand_with_nodes_and_edges() to avoid duplicates
    """
    processed_edges = set()
    type_node_edges = []
    entity_node_edges = []
    type_entity_edges = []
    graph_node_edges = []
    graph_engine = await get_graph_engine()

    for index, data_chunk in enumerate(data_chunks):
        graph = chunk_graphs[index]

        for node in graph.nodes:
            type_node_id = EntityType.id_for(node.type)
            entity_node_id = Entity.id_for(node.id)

            type_edge = (data_chunk.id, type_node_id, "exists_in")
            if type_edge not in processed_edges:
                type_node_edges.append(type_edge)
                processed_edges.add(type_edge)

            entity_edge = (data_chunk.id, entity_node_id, "mentioned_in")
            if entity_edge not in processed_edges:
                entity_node_edges.append(entity_edge)
                processed_edges.add(entity_edge)

            type_entity_edge = (entity_node_id, type_node_id, "is_a")
            if type_entity_edge not in processed_edges:
                type_entity_edges.append(type_entity_edge)
                processed_edges.add(type_entity_edge)

        chunk_graph_node_edges = [
            (
                Entity.id_for(edge.source_node_id),
                Entity.id_for(edge.target_node_id),
                generate_edge_name(edge.relationship_name),
            )
            for edge in graph.edges
        ]
        graph_node_edges.extend(chunk_graph_node_edges)

    candidate_edges = list(
        dict.fromkeys(
            [
                *type_node_edges,
                *entity_node_edges,
                *type_entity_edges,
                *graph_node_edges,
            ]
        )
    )
    existing_results = await graph_engine.has_edges(candidate_edges)

    # Most adapters return the subset of candidate triples that exists. Neo4j's
    # historical implementation returns one boolean per candidate instead, so
    # normalize both shapes at this boundary.
    if len(existing_results) == len(candidate_edges) and all(
        isinstance(result, bool) for result in existing_results
    ):
        existing_edges = [edge for edge, exists in zip(candidate_edges, existing_results) if exists]
    else:
        existing_edges = existing_results

    normalized_existing_edges = list(
        dict.fromkeys((str(edge[0]), str(edge[1]), str(edge[2])) for edge in existing_edges)
    )

    existing_edges_map = {}

    for edge in normalized_existing_edges:
        existing_edges_map[_create_edge_key(edge[0], edge[1], edge[2])] = True

    # expand_with_nodes_and_edges deliberately leaves existing relationships out
    # of the DataPoint graph, which prevents duplicate graph writes and vector
    # re-indexing. On graph-provenance graphs, preserve the later source's
    # ownership by attaching its ref directly to those relationships in one
    # batch. The run mapping makes this early attach rollback-safe if a later
    # pipeline task fails.
    if normalized_existing_edges and ctx is not None:
        provenance_kwargs = await graph_provenance_write_kwargs(graph_engine, ctx)
        source_ref_key = provenance_kwargs["source_ref_key"]
        if source_ref_key is not None:
            await graph_engine.attach_edge_source_refs(
                [EdgeIdentity(*edge) for edge in normalized_existing_edges],
                [source_ref_key],
                provenance_kwargs["pipeline_run_id"],
            )

    return existing_edges_map
