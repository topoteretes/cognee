from typing import AsyncGenerator, Dict, Any, List, Optional
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.modules.engine.utils import generate_node_id
from cognee.shared.logging_utils import get_logger
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Triplet
from cognee.tasks.storage import index_data_points

logger = get_logger("get_triplet_datapoints")


def _build_datapoint_type_index_mapping() -> Dict[str, List[str]]:
    """
    Build a mapping of DataPoint type names to their index_fields.

    Returns:
    --------
        - Dict[str, List[str]]: Mapping of type name to list of index field names
    """
    logger.debug("Building DataPoint type to index_fields mapping")
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
                    logger.debug(
                        f"Registered {subclass.__name__} with index_fields: {index_fields}"
                    )

    logger.info(
        f"Found {len(datapoint_type_index_property)} DataPoint types with index_fields: "
        f"{list(datapoint_type_index_property.keys())}"
    )
    return datapoint_type_index_property


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
            field_value = str(field_value).strip()

            if field_value:
                embeddable_values.append(field_value)

    return " ".join(embeddable_values) if embeddable_values else ""


def _extract_relationship_text(
    relationship: Dict[str, Any], datapoint_type_index_property: Dict[str, List[str]]
) -> str:
    """
    Extract relationship text from edge properties.

    Parameters:
    -----------
        - relationship (Dict[str, Any]): Dictionary containing relationship properties
        - datapoint_type_index_property (Dict[str, List[str]]): Mapping of type to index fields

    Returns:
    --------
        - str: Extracted relationship text or empty string
    """
    if not relationship:
        return ""

    edge_text = relationship.get("edge_text")
    if edge_text and isinstance(edge_text, str) and edge_text.strip():
        return edge_text.strip()

    # Fallback to extracting from EdgeType index_fields
    edge_type_index_fields = datapoint_type_index_property.get("EdgeType", [])
    return _extract_embeddable_text(relationship, edge_type_index_fields)


def _process_single_triplet(
    triplet_datapoint: Dict[str, Any],
    datapoint_type_index_property: Dict[str, List[str]],
    offset: int,
    idx: int,
) -> tuple[Optional[Triplet], Optional[str]]:
    """
    Process a single triplet and create a Triplet object.

    Parameters:
    -----------
        - triplet_datapoint (Dict[str, Any]): Raw triplet data from graph engine
        - datapoint_type_index_property (Dict[str, List[str]]): Type to index fields mapping
        - offset (int): Current batch offset
        - idx (int): Index within current batch

    Returns:
    --------
        - tuple[Optional[Triplet], Optional[str]]: (Triplet object, error message if skipped)
    """
    start_node = triplet_datapoint.get("start_node", {})
    end_node = triplet_datapoint.get("end_node", {})
    relationship = triplet_datapoint.get("relationship_properties", {})

    start_node_type = start_node.get("type")
    end_node_type = end_node.get("type")

    start_index_fields = datapoint_type_index_property.get(start_node_type, [])
    end_index_fields = datapoint_type_index_property.get(end_node_type, [])

    if not start_index_fields:
        logger.debug(
            f"No index_fields found for start_node type '{start_node_type}' in triplet {offset + idx}"
        )
    if not end_index_fields:
        logger.debug(
            f"No index_fields found for end_node type '{end_node_type}' in triplet {offset + idx}"
        )

    start_node_id = start_node.get("id", "")
    end_node_id = end_node.get("id", "")

    if not start_node_id or not end_node_id:
        return None, (
            f"Skipping triplet at offset {offset + idx}: missing node IDs "
            f"(start: {start_node_id}, end: {end_node_id})"
        )

    relationship_text = _extract_relationship_text(relationship, datapoint_type_index_property)
    start_node_text = _extract_embeddable_text(start_node, start_index_fields)
    end_node_text = _extract_embeddable_text(end_node, end_index_fields)

    if not start_node_text and not end_node_text and not relationship_text:
        return None, (
            f"Skipping triplet at offset {offset + idx}: empty embeddable text "
            f"(start_node_id: {start_node_id}, end_node_id: {end_node_id})"
        )

    embeddable_text = f"{start_node_text}-›{relationship_text}-›{end_node_text}".strip()

    relationship_name = relationship.get("relationship_name", "")
    triplet_id = generate_node_id(str(start_node_id) + str(relationship_name) + str(end_node_id))

    triplet_obj = Triplet(
        id=triplet_id, from_node_id=start_node_id, to_node_id=end_node_id, text=embeddable_text
    )

    return triplet_obj, None


async def get_triplet_datapoints(
    data,
    triplets_batch_size: int = 100,
) -> AsyncGenerator[Triplet, None]:
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
    if not data or data == [{}]:
        logger.info("Fetching graph data for current user")

    logger.info(f"Starting triplet datapoints extraction with batch size: {triplets_batch_size}")

    graph_engine = await get_graph_engine()
    graph_engine_type = type(graph_engine).__name__
    logger.debug(f"Using graph engine: {graph_engine_type}")

    if not hasattr(graph_engine, "get_triplets_batch"):
        error_msg = f"Graph adapter {graph_engine_type} does not support get_triplets_batch method"
        logger.error(error_msg)
        raise NotImplementedError(error_msg)

    datapoint_type_index_property = _build_datapoint_type_index_mapping()

    offset = 0
    total_triplets_processed = 0
    batch_number = 0

    while True:
        try:
            batch_number += 1
            logger.debug(
                f"Fetching triplet batch {batch_number} (offset: {offset}, limit: {triplets_batch_size})"
            )

            triplets_batch = await graph_engine.get_triplets_batch(
                offset=offset, limit=triplets_batch_size
            )

            if not triplets_batch:
                logger.info(f"No more triplets found at offset {offset}. Processing complete.")
                break

            logger.debug(f"Retrieved {len(triplets_batch)} triplets in batch {batch_number}")

            triplet_datapoints = []
            skipped_count = 0

            for idx, triplet_datapoint in enumerate(triplets_batch):
                try:
                    triplet_obj, error_msg = _process_single_triplet(
                        triplet_datapoint, datapoint_type_index_property, offset, idx
                    )

                    if error_msg:
                        logger.warning(error_msg)
                        skipped_count += 1
                        continue

                    if triplet_obj:
                        triplet_datapoints.append(triplet_obj)
                        yield triplet_obj

                except Exception as e:
                    logger.warning(
                        f"Error processing triplet at offset {offset + idx}: {e}. "
                        f"Skipping this triplet and continuing."
                    )
                    skipped_count += 1
                    continue

            if skipped_count > 0:
                logger.warning(
                    f"Skipped {skipped_count} out of {len(triplets_batch)} triplets in batch {batch_number}"
                )

            if not triplet_datapoints:
                logger.warning(
                    f"No valid triplet datapoints in batch {batch_number} after processing"
                )
                offset += len(triplets_batch)
                if len(triplets_batch) < triplets_batch_size:
                    break
                continue

            total_triplets_processed += len(triplet_datapoints)
            logger.info(
                f"Batch {batch_number} complete: processed {len(triplet_datapoints)} triplets "
                f"(total processed: {total_triplets_processed})"
            )

            offset += len(triplets_batch)
            if len(triplets_batch) < triplets_batch_size:
                logger.info(
                    f"Last batch retrieved (got {len(triplets_batch)} < {triplets_batch_size} triplets). "
                    f"Processing complete."
                )
                break

        except Exception as e:
            logger.error(
                f"Error retrieving triplet batch {batch_number} at offset {offset}: {e}",
                exc_info=True,
            )
            raise

    logger.info(
        f"Triplet datapoints extraction complete. "
        f"Processed {total_triplets_processed} triplets across {batch_number} batch(es)."
    )
