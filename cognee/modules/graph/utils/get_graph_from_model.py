from datetime import datetime, timezone
from typing import Tuple, List, Any, Dict, Optional
from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.storage.utils import copy_model


def _extract_field_data(field_value: Any) -> List[Tuple[Optional[Edge], List[DataPoint]]]:
    """Extract edge metadata and datapoints from a field value."""
    # Handle single DataPoint
    if isinstance(field_value, DataPoint):
        return [(None, [field_value])]

    # Handle list - could contain DataPoints, edge tuples, or mixed
    if isinstance(field_value, list) and len(field_value) > 0:
        result = []
        for item in field_value:
            # Handle tuple[Edge, DataPoint or list[DataPoint]]
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Edge):
                edge, data_value = item
                if isinstance(data_value, DataPoint):
                    result.append((edge, [data_value]))
                elif (
                    isinstance(data_value, list)
                    and len(data_value) > 0
                    and isinstance(data_value[0], DataPoint)
                ):
                    result.append((edge, data_value))
            # Handle single DataPoint in list
            elif isinstance(item, DataPoint):
                result.append((None, [item]))
        return result

    # Handle tuple[Edge, DataPoint or list[DataPoint]]
    if (
        isinstance(field_value, tuple)
        and len(field_value) == 2
        and isinstance(field_value[0], Edge)
    ):
        edge_metadata, data_value = field_value
        if isinstance(data_value, DataPoint):
            return [(edge_metadata, [data_value])]
        elif (
            isinstance(data_value, list)
            and len(data_value) > 0
            and isinstance(data_value[0], DataPoint)
        ):
            return [(edge_metadata, data_value)]

    # Regular property or empty list
    return []


def _create_edge_properties(
    source_id: str, target_id: str, relationship_name: str, edge_metadata: Optional[Edge]
) -> Dict[str, Any]:
    """Create edge properties dictionary with metadata if present."""
    properties = {
        "source_node_id": source_id,
        "target_node_id": target_id,
        "relationship_name": relationship_name,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    if edge_metadata:
        # Add edge metadata
        edge_data = edge_metadata.model_dump(exclude_none=True)
        properties.update(edge_data)

        # Add individual weights as separate fields for easier querying
        if edge_metadata.weights is not None:
            for weight_name, weight_value in edge_metadata.weights.items():
                properties[f"weight_{weight_name}"] = weight_value

    return properties


def _get_relationship_key(field_name: str, edge_metadata: Optional[Edge]) -> str:
    """Extract relationship key from edge metadata or use field name as fallback."""
    if (
        edge_metadata
        and hasattr(edge_metadata, "relationship_type")
        and edge_metadata.relationship_type
    ):
        return edge_metadata.relationship_type
    return field_name


def _generate_property_key(data_point_id: str, relationship_key: str, target_id: str) -> str:
    """Generate a unique property key for visited_properties tracking."""
    return f"{data_point_id}_{relationship_key}_{target_id}"


def _process_datapoint_field(
    data_point_id: str,
    field_name: str,
    edge_datapoint_pairs: List[Tuple[Optional[Edge], List[DataPoint]]],
    visited_properties: Dict[str, bool],
    properties_to_visit: set,
    excluded_properties: set,
) -> None:
    """Process a field containing DataPoints, always working with lists."""
    excluded_properties.add(field_name)

    for edge_metadata, datapoints in edge_datapoint_pairs:
        relationship_key = _get_relationship_key(field_name, edge_metadata)

        for datapoint in datapoints:
            property_key = _generate_property_key(
                data_point_id, relationship_key, str(datapoint.id)
            )
            if property_key in visited_properties:
                continue

            # Always use field_name since we're working with lists
            properties_to_visit.add(field_name)


def _targets_generator(
    data_point: DataPoint,
    properties_to_visit: set,
) -> Tuple[DataPoint, str, Optional[Edge]]:
    """Generator that yields (target_datapoint, field_name, edge_metadata) tuples."""
    for field_name in properties_to_visit:
        field_value = getattr(data_point, field_name)
        edge_datapoint_pairs = _extract_field_data(field_value)

        if not edge_datapoint_pairs:
            continue

        for edge_metadata, datapoints in edge_datapoint_pairs:
            for target_datapoint in datapoints:
                yield target_datapoint, field_name, edge_metadata


async def get_graph_from_model(
    data_point: DataPoint,
    added_nodes: Dict[str, bool],
    added_edges: Dict[str, bool],
    visited_properties: Optional[Dict[str, bool]] = None,
    include_root: bool = True,
) -> Tuple[List[DataPoint], List[Tuple[str, str, str, Dict[str, Any]]]]:
    """
    Extract graph representation from a DataPoint model.

    Args:
        data_point: The DataPoint to extract graph from
        added_nodes: Dictionary tracking already processed nodes
        added_edges: Dictionary tracking already processed edges
        visited_properties: Dictionary tracking visited properties to avoid cycles
        include_root: Whether to include the root node in results

    Returns:
        Tuple of (nodes, edges) extracted from the model
    """
    if str(data_point.id) in added_nodes:
        return [], []

    nodes = []
    edges = []
    visited_properties = visited_properties or {}
    data_point_id = str(data_point.id)

    data_point_properties = {"type": type(data_point).__name__}
    excluded_properties = set()
    properties_to_visit = set()

    # Analyze all fields to categorize them as properties or relationships
    for field_name, field_value in data_point:
        if field_name == "metadata":
            continue

        edge_datapoint_pairs = _extract_field_data(field_value)

        if not edge_datapoint_pairs:
            # Regular property
            data_point_properties[field_name] = field_value
        else:
            # DataPoint relationship
            _process_datapoint_field(
                data_point_id,
                field_name,
                edge_datapoint_pairs,
                visited_properties,
                properties_to_visit,
                excluded_properties,
            )

    # Create node for current DataPoint if needed
    if include_root and data_point_id not in added_nodes:
        SimpleDataPointModel = copy_model(
            type(data_point), exclude_fields=list(excluded_properties)
        )
        nodes.append(SimpleDataPointModel(**data_point_properties))
        added_nodes[data_point_id] = True

    # Process all relationships using generator
    for target_datapoint, field_name, edge_metadata in _targets_generator(
        data_point, properties_to_visit
    ):
        relationship_name = _get_relationship_key(field_name, edge_metadata)

        # Create edge if not already added
        edge_key = f"{data_point_id}_{target_datapoint.id}_{field_name}"
        if edge_key not in added_edges:
            edge_properties = _create_edge_properties(
                data_point.id, target_datapoint.id, relationship_name, edge_metadata
            )
            edges.append((data_point.id, target_datapoint.id, relationship_name, edge_properties))
            added_edges[edge_key] = True

        # Mark property as visited - CRITICAL for preventing infinite loops
        property_key = _generate_property_key(
            data_point_id, relationship_name, str(target_datapoint.id)
        )
        visited_properties[property_key] = True

        # Recursively process target node if not already processed
        if str(target_datapoint.id) in added_nodes:
            continue

        child_nodes, child_edges = await get_graph_from_model(
            target_datapoint,
            include_root=True,
            added_nodes=added_nodes,
            added_edges=added_edges,
            visited_properties=visited_properties,
        )
        nodes.extend(child_nodes)
        edges.extend(child_edges)

    return nodes, edges


def get_own_property_nodes(
    property_nodes: List[DataPoint], property_edges: List[Tuple[str, str, str, Dict[str, Any]]]
) -> List[DataPoint]:
    """
    Filter nodes to return only those that are not destinations of any edges.

    Args:
        property_nodes: List of all nodes
        property_edges: List of all edges

    Returns:
        List of nodes that are not edge destinations
    """
    destination_node_ids = {str(edge[1]) for edge in property_edges}
    return [node for node in property_nodes if str(node.id) not in destination_node_ids]
