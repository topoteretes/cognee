import asyncio

from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.shared.logging_utils import get_logger
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple, Union
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData

logger = get_logger()


async def index_graph_edges(
    edges_data: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]] = None,
):
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

        if edges_data is None:
            graph_engine = await get_graph_engine()
            _, edges_data = await graph_engine.get_graph_data()
            logger.warning(
                "Your graph edge embedding is deprecated, please pass edges to the index_graph_edges directly."
            )
    except Exception as e:
        logger.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    edge_types = Counter(
        item.get("relationship_name")
        for edge in edges_data
        for item in edge
        if isinstance(item, dict) and "relationship_name" in item
    )

    for text, count in edge_types.items():
        edge = EdgeType(
            id=generate_edge_id(edge_id=text), relationship_name=text, number_of_edges=count
        )
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

    # Get maximum batch size for embedding model
    batch_size = vector_engine.embedding_engine.get_batch_size()
    tasks: list[asyncio.Task] = []

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")

        # Create embedding tasks to run in parallel later
        for start in range(0, len(indexable_points), batch_size):
            batch = indexable_points[start : start + batch_size]

            tasks.append(vector_engine.index_data_points(index_name, field_name, batch))

    # Start all embedding tasks and wait for completion
    await asyncio.gather(*tasks)

    return None
