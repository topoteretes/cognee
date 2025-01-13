import logging
from collections import Counter

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.tasks.temporal_awareness.graphiti_model import GraphitiNode


async def index_graphiti_nodes_and_edges():
    try:
        created_indexes = {}
        index_points = {}

        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as e:
        logging.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    nodes_data, edges_data = await graph_engine.get_model_independent_graph_data()

    for node_data in nodes_data[0]["nodes"]:
        graphiti_node = GraphitiNode(
            **{key: node_data[key] for key in ("content", "name", "summary") if key in node_data}
        )

        data_point_type = type(graphiti_node)

        for field_name in graphiti_node._metadata["index_fields"]:
            index_name = f"{data_point_type.__tablename__}.{field_name}"

            if index_name not in created_indexes:
                await vector_engine.create_vector_index(data_point_type.__tablename__, field_name)
                created_indexes[index_name] = True

            if index_name not in index_points:
                index_points[index_name] = []

            if getattr(graphiti_node, field_name, None) is not None:
                indexed_data_point = graphiti_node.model_copy()
                indexed_data_point._metadata["index_fields"] = [field_name]
                index_points[index_name].append(indexed_data_point)

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")
        await vector_engine.index_data_points(index_name, field_name, indexable_points)

    edge_types = Counter(
        edge[1]
        for edge in edges_data[0]["relationships"]
        if isinstance(edge, tuple) and len(edge) == 3
    )

    for text, count in edge_types.items():
        edge = EdgeType(relationship_name=text, number_of_edges=count)
        data_point_type = type(edge)

        for field_name in edge._metadata["index_fields"]:
            index_name = f"{data_point_type.__tablename__}.{field_name}"

            if index_name not in created_indexes:
                await vector_engine.create_vector_index(data_point_type.__tablename__, field_name)
                created_indexes[index_name] = True

            if index_name not in index_points:
                index_points[index_name] = []

            indexed_data_point = edge.model_copy()
            indexed_data_point._metadata["index_fields"] = [field_name]
            index_points[index_name].append(indexed_data_point)

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")
        await vector_engine.index_data_points(index_name, field_name, indexable_points)

    return None


async def index_graph_edges():
    """
    Indexes graph edges by creating and managing vector indexes for relationship types.

    This function retrieves edge data from the graph engine, counts distinct relationship
    types, and creates `EdgeType` pydantic objects. It ensures that vector indexes are created for
    the `relationship_name` field.

    Steps:
    1. Initialize the vector engine and graph engine.
    2. Retrieve graph edge data and count relationship types (`relationship_name`).
    3. Create vector indexes for `relationship_name` if they don't exist.
    4. Transform the counted relationships into `EdgeType` objects.
    5. Index the transformed data points in the vector engine.

    Raises:
        RuntimeError: If initialization of the vector engine or graph engine fails.

    Returns:
        None
    """
    try:
        created_indexes = {}
        index_points = {}

        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as e:
        logging.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    _, edges_data = await graph_engine.get_graph_data()

    edge_types = Counter(
        item.get("relationship_name")
        for edge in edges_data
        for item in edge
        if isinstance(item, dict) and "relationship_name" in item
    )

    for text, count in edge_types.items():
        edge = EdgeType(relationship_name=text, number_of_edges=count)
        data_point_type = type(edge)

        for field_name in edge._metadata["index_fields"]:
            index_name = f"{data_point_type.__tablename__}.{field_name}"

            if index_name not in created_indexes:
                await vector_engine.create_vector_index(data_point_type.__tablename__, field_name)
                created_indexes[index_name] = True

            if index_name not in index_points:
                index_points[index_name] = []

            indexed_data_point = edge.model_copy()
            indexed_data_point._metadata["index_fields"] = [field_name]
            index_points[index_name].append(indexed_data_point)

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")
        await vector_engine.index_data_points(index_name, field_name, indexable_points)

    return None
