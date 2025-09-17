from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.utils import generate_node_id
from cognee.shared.data_models import KnowledgeGraph


async def retrieve_existing_edges(
    data_chunks: list[DataPoint],
    chunk_graphs: list[KnowledgeGraph],
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

    Returns:
        dict[str, bool]: A mapping of edge keys to boolean values indicating existence.
            Edge keys are formatted as concatenated strings: "{source_id}{target_id}{relationship_name}".
            All values in the returned dictionary are True (indicating the edge exists).

    Note:
        - The function generates several types of edges for checking:
          * Type node edges: (chunk_id, type_node_id, "exists_in")
          * Entity node edges: (chunk_id, entity_node_id, "mentioned_in")
          * Type-entity edges: (entity_node_id, type_node_id, "is_a")
          * Graph node edges: extracted from the knowledge graph relationships
        - Uses generate_node_id() to ensure consistent node ID formatting
        - Prevents processing the same node multiple times using a processed_nodes tracker
        - The returned mapping can be used with expand_with_nodes_and_edges() to avoid duplicates
    """
    processed_nodes = {}
    type_node_edges = []
    entity_node_edges = []
    type_entity_edges = []
    graph_engine = await get_graph_engine()

    for index, data_chunk in enumerate(data_chunks):
        graph = chunk_graphs[index]

        for node in graph.nodes:
            type_node_id = generate_node_id(node.type)
            entity_node_id = generate_node_id(node.id)

            if str(type_node_id) not in processed_nodes:
                type_node_edges.append((data_chunk.id, type_node_id, "exists_in"))
                processed_nodes[str(type_node_id)] = True

            if str(entity_node_id) not in processed_nodes:
                entity_node_edges.append((data_chunk.id, entity_node_id, "mentioned_in"))
                type_entity_edges.append((entity_node_id, type_node_id, "is_a"))
                processed_nodes[str(entity_node_id)] = True

        graph_node_edges = [
            (edge.target_node_id, edge.source_node_id, edge.relationship_name)
            for edge in graph.edges
        ]

    existing_edges = await graph_engine.has_edges(
        [
            *type_node_edges,
            *entity_node_edges,
            *type_entity_edges,
            *graph_node_edges,
        ]
    )

    existing_edges_map = {}

    for edge in existing_edges:
        existing_edges_map[str(edge[0]) + str(edge[1]) + edge[2]] = True

    return existing_edges_map
