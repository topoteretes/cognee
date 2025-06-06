from cognee.shared.logging_utils import get_logger, ERROR
from collections import Counter

from cognee.tasks.temporal_awareness.graphiti_model import GraphitiNode
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.models.EdgeType import EdgeType

logger = get_logger(level=ERROR)


async def index_and_transform_graphiti_nodes_and_edges():
    try:
        created_indexes = {}
        index_points = {}

        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as e:
        logger.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    await graph_engine.query("""MATCH (n) SET n.id = n.uuid RETURN n""", params={})
    await graph_engine.query(
        """MATCH (source)-[r]->(target) SET r.source_node_id = source.id,
                             r.target_node_id = target.id,
                             r.relationship_name = type(r) RETURN r""",
        params={},
    )
    await graph_engine.query(
        """MATCH (n) SET n.text = COALESCE(n.summary, n.content) RETURN n""", params={}
    )

    nodes_data, edges_data = await graph_engine.get_graph_data()

    for node_id, node_data in nodes_data:
        graphiti_node = GraphitiNode(
            **{key: node_data[key] for key in ("content", "name", "summary") if key in node_data},
            id=node_id,
        )

        data_point_type = type(graphiti_node)

        for field_name in graphiti_node.metadata["index_fields"]:
            index_name = f"{data_point_type.__name__}.{field_name}"

            if index_name not in created_indexes:
                await vector_engine.create_vector_index(data_point_type.__name__, field_name)
                created_indexes[index_name] = True

            if index_name not in index_points:
                index_points[index_name] = []

            if getattr(graphiti_node, field_name, None) is not None:
                indexed_data_point = graphiti_node.model_copy()
                indexed_data_point.metadata["index_fields"] = [field_name]
                index_points[index_name].append(indexed_data_point)

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")
        await vector_engine.index_data_points(index_name, field_name, indexable_points)

    edge_types = Counter(
        edge[2]  # The edge key (relationship name) is at index 2
        for edge in edges_data
    )

    for text, count in edge_types.items():
        edge = EdgeType(relationship_name=text, number_of_edges=count)
        data_point_type = type(edge)

        for field_name in edge.metadata["index_fields"]:
            index_name = f"{data_point_type.__name__}.{field_name}"

            if index_name not in created_indexes:
                await vector_engine.create_vector_index(data_point_type.__name__, field_name)
                created_indexes[index_name] = True

            if index_name not in index_points:
                index_points[index_name] = []

            indexed_data_point = edge.model_copy()
            indexed_data_point.metadata["index_fields"] = [field_name]
            index_points[index_name].append(indexed_data_point)

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")
        await vector_engine.index_data_points(index_name, field_name, indexable_points)

    return None
