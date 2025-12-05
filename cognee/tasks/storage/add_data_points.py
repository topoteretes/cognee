import asyncio
from typing import List, Dict, Optional
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from .index_data_points import index_data_points
from .index_graph_edges import index_graph_edges
from cognee.modules.engine.models import Triplet
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.exceptions import (
    InvalidDataPointsInAddDataPointsError,
)
from ...modules.engine.utils import generate_node_id

logger = get_logger("add_data_points")


async def add_data_points(
    data_points: List[DataPoint], custom_edges: Optional[List] = None, embed_triplets: bool = False
) -> List[DataPoint]:
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
        custom_edges (List[tuple]): Custom edges between datapoints.
        embed_triplets (bool):
            If True, creates and indexes triplet embeddings from the graph structure.
            Defaults to False.

    Returns:
        List[DataPoint]:
            The original list of data points after processing and insertion.

    Side Effects:
        - Calls `get_graph_from_model` concurrently for each data point.
        - Deduplicates nodes and edges across all results.
        - Updates the node index via `index_data_points`.
        - Inserts nodes and edges into the graph engine.
        - Optionally updates the edge index via `index_graph_edges`.
        - Optionally creates and indexes triplet embeddings if embed_triplets is True.
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

    if isinstance(custom_edges, list) and custom_edges:
        # This must be handled separately from datapoint edges, created a task in linear to dig deeper but (COG-3488)
        await graph_engine.add_edges(custom_edges)
        await index_graph_edges(custom_edges)
        edges.extend(custom_edges)

    if embed_triplets:
        triplets = _create_triplets_from_graph(nodes, edges)
        if triplets:
            await index_data_points(triplets)
            logger.info(f"Created and indexed {len(triplets)} triplets from graph structure")

    return data_points


def _extract_embeddable_text_from_datapoint(data_point: DataPoint) -> str:
    """
    Extract embeddable text from a DataPoint using its index_fields metadata.
    Uses the same approach as index_data_points.

    Parameters:
    -----------
        - data_point (DataPoint): The data point to extract text from.

    Returns:
    --------
        - str: Concatenated string of all embeddable property values, or empty string if none found.
    """
    if not data_point or not hasattr(data_point, "metadata"):
        return ""

    index_fields = data_point.metadata.get("index_fields", [])
    if not index_fields:
        return ""

    embeddable_values = []
    for field_name in index_fields:
        field_value = getattr(data_point, field_name, None)
        if field_value is not None:
            field_value = str(field_value).strip()

            if field_value:
                embeddable_values.append(field_value)

    return " ".join(embeddable_values) if embeddable_values else ""


def _create_triplets_from_graph(nodes: List[DataPoint], edges: List[tuple]) -> List[Triplet]:
    """
    Create Triplet objects from graph nodes and edges.

    This function processes graph edges and their corresponding nodes to create
    triplet datapoints with embeddable text, similar to the triplet embeddings pipeline.

    Parameters:
    -----------
        - nodes (List[DataPoint]): List of graph nodes extracted from data points
        - edges (List[tuple]): List of edge tuples in format
          (source_node_id, target_node_id, relationship_name, properties_dict)
          Note: All edges including those from DocumentChunk.contains are already extracted
          by get_graph_from_model and included in this list.

    Returns:
    --------
        - List[Triplet]: List of Triplet objects ready for indexing
    """
    node_map: Dict[str, DataPoint] = {}
    for node in nodes:
        if hasattr(node, "id"):
            node_id = str(node.id)
            if node_id not in node_map:
                node_map[node_id] = node

    triplets = []
    skipped_count = 0
    seen_ids = set()

    for edge_tuple in edges:
        if len(edge_tuple) < 4:
            continue

        source_node_id, target_node_id, relationship_name, edge_properties = (
            edge_tuple[0],
            edge_tuple[1],
            edge_tuple[2],
            edge_tuple[3],
        )

        source_node = node_map.get(str(source_node_id))
        target_node = node_map.get(str(target_node_id))

        if not source_node or not target_node or relationship_name is None:
            skipped_count += 1
            continue

        source_node_text = _extract_embeddable_text_from_datapoint(source_node)
        target_node_text = _extract_embeddable_text_from_datapoint(target_node)

        relationship_text = ""
        if isinstance(edge_properties, dict):
            edge_text = edge_properties.get("edge_text")
            if edge_text and isinstance(edge_text, str) and edge_text.strip():
                relationship_text = edge_text.strip()

        if not relationship_text and relationship_name:
            relationship_text = relationship_name

        if not source_node_text and not relationship_text and not relationship_name:
            skipped_count += 1
            continue

        embeddable_text = f"{source_node_text} -› {relationship_text}-›{target_node_text}".strip()

        triplet_id = generate_node_id(str(source_node_id) + relationship_name + str(target_node_id))

        if triplet_id in seen_ids:
            continue
        seen_ids.add(triplet_id)

        triplets.append(
            Triplet(
                id=triplet_id,
                from_node_id=str(source_node_id),
                to_node_id=str(target_node_id),
                text=embeddable_text,
            )
        )

    return triplets
