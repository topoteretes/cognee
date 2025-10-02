import asyncio
from typing import List
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from .index_data_points import index_data_points
from .index_graph_edges import index_graph_edges
from cognee.tasks.storage.exceptions import (
    InvalidDataPointsInAddDataPointsError,
)


async def add_data_points(data_points: List[DataPoint]) -> List[DataPoint]:
    """
    Add a batch of data points to the graph database by extracting nodes and edges,
    deduplicating them, and indexing them for retrieval.

    This function parallelizes the graph extraction for each data point,
    merges the resulting nodes and edges, and ensures uniqueness before
    committing them to the underlying graph engine. It also updates the
    associated retrieval indices for nodes and (optionally) edges.

    Args:
        data_points (List[DataPoint]):
            A list of data points to process and insert into the graph.

    Returns:
        List[DataPoint]:
            The original list of data points after processing and insertion.

    Side Effects:
        - Calls `get_graph_from_model` concurrently for each data point.
        - Deduplicates nodes and edges across all results.
        - Updates the node index via `index_data_points`.
        - Inserts nodes and edges into the graph engine.
        - Optionally updates the edge index via `index_graph_edges`.
    """

    if not isinstance(data_points, list):
        raise InvalidDataPointsInAddDataPointsError("data_points must be a list.")
    if not all(isinstance(dp, DataPoint) for dp in data_points):
        raise InvalidDataPointsInAddDataPointsError("data_points: each item must be a DataPoint.")

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

    graph_engine = await get_graph_engine()

    await graph_engine.add_nodes(nodes)
    await index_data_points(nodes)

    await graph_engine.add_edges(edges)
    await index_graph_edges(edges)

    return data_points
