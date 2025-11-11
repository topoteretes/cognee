from collections import Counter
from typing import Optional, Dict, Any, List, Tuple, Union

from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData
from cognee.tasks.storage.index_data_points import index_data_points

logger = get_logger()


def _get_edge_text(item: dict) -> str:
    """Extract edge text for embedding - prefers edge_text field with fallback."""
    if "edge_text" in item:
        return item["edge_text"]

    if "relationship_name" in item:
        return item["relationship_name"]

    return ""


def create_edge_type_datapoints(edges_data) -> list[EdgeType]:
    """Transform raw edge data into EdgeType datapoints."""
    edge_texts = [
        _get_edge_text(item)
        for edge in edges_data
        for item in edge
        if isinstance(item, dict) and "relationship_name" in item
    ]

    edge_types = Counter(edge_texts)

    return [
        EdgeType(id=generate_edge_id(edge_id=text), relationship_name=text, number_of_edges=count)
        for text, count in edge_types.items()
    ]


async def index_graph_edges(
    edges_data: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]] = None,
):
    """
    Indexes graph edges by creating and managing vector indexes for relationship types.

    This function retrieves edge data from the graph engine, counts distinct relationship
    types, and creates `EdgeType` pydantic objects. It ensures that vector indexes are created for
    the `relationship_name` field.

    Steps:
    1. Initialize the graph engine if needed and retrieve edge data.
    2. Transform edge data into EdgeType datapoints.
    3. Index the EdgeType datapoints using the standard indexing function.

    Raises:
        RuntimeError: If initialization of the graph engine fails.

    Returns:
        None
    """
    try:
        if edges_data is None:
            graph_engine = await get_graph_engine()
            _, edges_data = await graph_engine.get_graph_data()
            logger.warning(
                "Your graph edge embedding is deprecated, please pass edges to the index_graph_edges directly."
            )
    except Exception as e:
        logger.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    edge_type_datapoints = create_edge_type_datapoints(edges_data)
    await index_data_points(edge_type_datapoints)

    return None
