from typing import Any, Dict, List, Optional, Tuple, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData
from cognee.modules.graph.utils.edge_index_points import build_edge_index_points
from cognee.tasks.storage.index_data_points import index_data_points

logger = get_logger()


async def index_graph_edges(
    edges_data: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]] = None,
    vector_engine=None,
    graph_engine=None,
):
    """
    Index graph edge types and individual edge prose for vector retrieval.

    Edge types are keyed by structural relationship name and use current graph
    counts. Edge instances retain their per-edge retrieval prose.

    Raises:
        RuntimeError: If initialization of the graph engine fails.

    Returns:
        None
    """
    try:
        if graph_engine is None:
            graph_engine = await get_graph_engine()

        if edges_data is None:
            _, edges_data = await graph_engine.get_graph_data()
            logger.warning(
                "Your graph edge embedding is deprecated, please pass edges to the index_graph_edges directly."
            )

        local_points = build_edge_index_points(edges_data)
        relationship_names = [edge_type.relationship_name for edge_type in local_points.edge_types]
        relationship_counts = await graph_engine.get_edge_type_counts(relationship_names)
        edge_points = build_edge_index_points(edges_data, relationship_counts=relationship_counts)

        await index_data_points(
            edge_points.edge_types + edge_points.edge_instances,
            vector_engine=vector_engine,
        )
    except Exception as e:
        logger.error("Failed to index graph edges: %s", e)
        raise RuntimeError("Graph edge indexing error") from e

    return None
