from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.utils import generate_node_id
from cognee.shared.data_models import KnowledgeGraph


async def retrieve_existing_edges(
    graph_node_index: list[tuple[DataPoint, KnowledgeGraph]],
    graph_engine: GraphDBInterface,
) -> dict[str, bool]:
    processed_nodes = {}
    type_node_edges = []
    entity_node_edges = []
    type_entity_edges = []

    for graph_source, graph in graph_node_index:
        for node in graph.nodes:
            type_node_id = generate_node_id(node.type)
            entity_node_id = generate_node_id(node.id)

            if str(type_node_id) not in processed_nodes:
                type_node_edges.append(
                    (str(graph_source), str(type_node_id), "exists_in")
                )
                processed_nodes[str(type_node_id)] = True

            if str(entity_node_id) not in processed_nodes:
                entity_node_edges.append(
                    (str(graph_source), entity_node_id, "mentioned_in")
                )
                type_entity_edges.append(
                    (str(entity_node_id), str(type_node_id), "is_a")
                )
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
        existing_edges_map[edge[0] + edge[1] + edge[2]] = True

    return existing_edges_map
