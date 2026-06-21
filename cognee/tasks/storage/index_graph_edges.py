from collections import Counter
from typing import Optional, Dict, Any, List, Tuple, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData
from cognee.tasks.storage.index_data_points import index_data_points

logger = get_logger()


def _get_edge_properties(edge) -> dict:
    if isinstance(edge, dict):
        return edge

    for item in reversed(edge):
        if isinstance(item, dict):
            return item

    return {}


def _get_edge_relationship_name(edge, properties: dict) -> str:
    if isinstance(edge, (list, tuple)) and len(edge) >= 3 and not isinstance(edge[2], dict):
        return edge[2]

    return properties.get("relationship_name", "")


def _get_edge_text(edge) -> str:
    """Extract nonblank edge retrieval text with relationship_name fallback."""
    properties = _get_edge_properties(edge)
    relationship_name = _get_edge_relationship_name(edge, properties)
    return get_edge_retrieval_text(properties.get("edge_text"), relationship_name)


def create_edge_type_datapoints(edges_data) -> list[EdgeType]:
    """Transform raw edge data into EdgeType datapoints."""
    edge_texts = []
    for edge in edges_data:
        edge_text = _get_edge_text(edge)
        if edge_text:
            edge_texts.append(edge_text)

    edge_types = Counter(edge_texts)

    return [
        EdgeType(relationship_name=text, number_of_edges=count)
        for text, count in edge_types.items()
    ]


async def index_graph_edges(
    edges_data: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]] = None,
    vector_engine=None,
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

        edge_type_datapoints = create_edge_type_datapoints(edges_data)

        await index_data_points(edge_type_datapoints, vector_engine=vector_engine)
    except Exception as e:
        logger.error("Failed to index graph edges: %s", e)
        raise RuntimeError("Graph edge indexing error") from e

    return None
