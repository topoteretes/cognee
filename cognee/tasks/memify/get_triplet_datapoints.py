"""Task to get triplet datapoints from the graph database as an async generator."""

from typing import AsyncGenerator, Dict, Any, List, Optional
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.shared.logging_utils import get_logger
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Triplet
from cognee.tasks.storage import add_data_points

logger = get_logger()


def _extract_embeddable_text(node_or_edge: Dict[str, Any], index_fields: List[str]) -> str:
    """
    Extract and concatenate embeddable properties from a node or edge dictionary.
    
    Parameters:
    -----------
        - node_or_edge (Dict[str, Any]): Dictionary containing node or edge properties.
        - index_fields (List[str]): List of field names to extract and concatenate.
    
    Returns:
    --------
        - str: Concatenated string of all embeddable property values, or empty string if none found.
    """
    if not node_or_edge or not index_fields:
        return ""
    
    embeddable_values = []
    for field_name in index_fields:
        field_value = node_or_edge.get(field_name)
        if field_value is not None:
            if isinstance(field_value, str):
                field_value = field_value.strip()
            else:
                field_value = str(field_value).strip()
            
            if field_value:
                embeddable_values.append(field_value)
    
    return " ".join(embeddable_values) if embeddable_values else ""


async def get_triplet_datapoints(
    triplets_batch_size: int = 100,
) -> AsyncGenerator[List[Dict[str, Any]], None]:
    """
    Async generator that yields batches of triplet datapoints with embeddable text extracted.
    
    Each triplet in the batch includes:
    - Original triplet structure (start_node, relationship_properties, end_node)
    - Extracted embeddable text for each element based on index_fields
    
    Parameters:
    -----------
        - triplets_batch_size (int): Number of triplets to retrieve per batch. Default is 100.
    
    Yields:
    -------
        - List[Dict[str, Any]]: A batch of triplets, each enriched with embeddable text.
    """
    graph_engine = await get_graph_engine()

    if not hasattr(graph_engine, "get_triplets_batch"):
        raise NotImplementedError(
            f"Graph adapter {type(graph_engine).__name__} does not support get_triplets_batch method"
        )

    # Build mapping of DataPoint type names to their index_fields
    subclasses = get_all_subclasses(DataPoint)
    datapoint_type_index_property = {}

    for subclass in subclasses:
        if "metadata" in subclass.model_fields:
            metadata_field = subclass.model_fields["metadata"]
            default = getattr(metadata_field, "default", None)
            if isinstance(default, dict):
                index_fields = default.get("index_fields", [])
                if index_fields:
                    datapoint_type_index_property[subclass.__name__] = index_fields

    offset = 0
    while True:
        try:
            triplets_batch = await graph_engine.get_triplets_batch(
                offset=offset, limit=triplets_batch_size
            )
            
            if not triplets_batch:
                break

            triplet_datapoints = []
            for triplet_datapoint in triplets_batch:
                start_node = triplet_datapoint.get("start_node", {})
                end_node = triplet_datapoint.get("end_node", {})
                relationship = triplet_datapoint.get("relationship_properties", {})
                

                start_node_type = start_node.get("type")
                end_node_type = end_node.get("type")
                

                start_index_fields = datapoint_type_index_property.get(start_node_type, [])
                end_index_fields = datapoint_type_index_property.get(end_node_type, [])
                

                relationship_text = ""
                if relationship:
                    edge_text = relationship.get("edge_text")
                    if edge_text and isinstance(edge_text, str) and edge_text.strip():
                        relationship_text = edge_text.strip()
                    else:
                        edge_type_index_fields = datapoint_type_index_property.get("EdgeType", [])
                        relationship_text = _extract_embeddable_text(relationship, edge_type_index_fields)

                start_node_text = _extract_embeddable_text(start_node, start_index_fields)
                end_node_text = _extract_embeddable_text(end_node, end_index_fields)

                start_node_id = start_node.get("id", '')
                end_node_id = end_node.get("id", '')
                embeddable_text = start_node_text + " " + relationship_text + " " + end_node_text
                
                triplet_obj = Triplet(from_node_id=start_node_id, to_node_id=end_node_id, text=embeddable_text)
                
                triplet_datapoints.append(triplet_obj)

            await add_data_points(triplet_datapoints)

            yield triplet_datapoints

            offset += len(triplets_batch)
            if len(triplets_batch) < triplets_batch_size:
                break

        except Exception as e:
            logger.error(f"Error retrieving triplet batch at offset {offset}: {e}")
            raise

