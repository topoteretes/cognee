from datetime import datetime, timezone
from typing import Tuple, List, Any, Dict, Optional
from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.storage.utils import copy_model


def _extract_field_info(field_value: Any) -> Tuple[str, Any, Optional[Edge]]:
    """Extract field type, actual value, and edge metadata from a field value."""

    # Handle tuple[Edge, DataPoint]
    if (
        isinstance(field_value, tuple)
        and len(field_value) == 2
        and isinstance(field_value[0], Edge)
        and isinstance(field_value[1], DataPoint)
    ):
        return "single_datapoint_with_edge", field_value[1], field_value[0]

    # Handle tuple[Edge, list[DataPoint]]
    if (
        isinstance(field_value, tuple)
        and len(field_value) == 2
        and isinstance(field_value[0], Edge)
        and isinstance(field_value[1], list)
        and len(field_value[1]) > 0
        and isinstance(field_value[1][0], DataPoint)
    ):
        return "list_datapoint_with_edge", field_value[1], field_value[0]

    # Handle single DataPoint
    if isinstance(field_value, DataPoint):
        return "single_datapoint", field_value, None

    # Handle list of DataPoints
    if (
        isinstance(field_value, list)
        and len(field_value) > 0
        and isinstance(field_value[0], DataPoint)
    ):
        return "list_datapoint", field_value, None

    # Regular property
    return "property", field_value, None


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

    data_point_properties = {"type": type(data_point).__name__}
    excluded_properties = set()
    properties_to_visit = set()  # Use set like original

    for field_name, field_value in data_point:
        if field_name == "metadata":
            continue

        field_type, actual_value, edge_metadata = _extract_field_info(field_value)

        if field_type == "property":
            data_point_properties[field_name] = field_value
        elif field_type in ["single_datapoint", "single_datapoint_with_edge"]:
            excluded_properties.add(field_name)

            # Use relationship_type if available, otherwise use field_name
            relationship_key = field_name
            if (
                edge_metadata
                and hasattr(edge_metadata, "relationship_type")
                and edge_metadata.relationship_type
            ):
                relationship_key = edge_metadata.relationship_type

            property_key = str(data_point.id) + relationship_key + str(actual_value.id)
            if property_key in visited_properties:
                continue

            properties_to_visit.add(field_name)
        elif field_type in ["list_datapoint", "list_datapoint_with_edge"]:
            excluded_properties.add(field_name)

            # Use relationship_type if available, otherwise use field_name
            relationship_key = field_name
            if (
                edge_metadata
                and hasattr(edge_metadata, "relationship_type")
                and edge_metadata.relationship_type
            ):
                relationship_key = edge_metadata.relationship_type

            for index, item in enumerate(actual_value):
                property_key = str(data_point.id) + relationship_key + str(item.id)
                if property_key in visited_properties:
                    continue

                properties_to_visit.add(f"{field_name}.{index}")

    # Create node for current DataPoint if needed
    if include_root and str(data_point.id) not in added_nodes:
        SimpleDataPointModel = copy_model(
            type(data_point), exclude_fields=list(excluded_properties)
        )
        nodes.append(SimpleDataPointModel(**data_point_properties))
        added_nodes[str(data_point.id)] = True

    # Process all relationships
    for field_name_with_index in properties_to_visit:
        index = None
        field_name = field_name_with_index

        if "." in field_name_with_index:
            field_name, index = field_name_with_index.split(".")

        field_value = getattr(data_point, field_name)

        # Extract edge metadata if field_value is a tuple with Edge metadata
        edge_metadata = None
        if (
            isinstance(field_value, tuple)
            and len(field_value) == 2
            and isinstance(field_value[0], Edge)
        ):
            edge_metadata, field_value = field_value

        if index is not None:
            field_value = field_value[int(index)]

        edge_key = str(data_point.id) + str(field_value.id) + field_name

        if str(edge_key) not in added_edges:
            # Determine relationship name: use relationship_type from edge metadata if available, else field_name
            relationship_name = field_name
            if (
                edge_metadata
                and hasattr(edge_metadata, "relationship_type")
                and edge_metadata.relationship_type
            ):
                relationship_name = edge_metadata.relationship_type

            # Build edge properties with weights support
            edge_properties = _create_edge_properties(
                data_point.id, field_value.id, relationship_name, edge_metadata
            )
            edges.append((data_point.id, field_value.id, relationship_name, edge_properties))
            added_edges[str(edge_key)] = True

        if str(field_value.id) in added_nodes:
            continue

        property_nodes, property_edges = await get_graph_from_model(
            field_value,
            include_root=True,
            added_nodes=added_nodes,
            added_edges=added_edges,
            visited_properties=visited_properties,
        )

        for node in property_nodes:
            nodes.append(node)

        for edge in property_edges:
            edges.append(edge)

        # Mark property as visited - CRITICAL for preventing infinite loops
        # Use the same relationship_key logic as in discovery phase
        relationship_key = field_name
        if (
            edge_metadata
            and hasattr(edge_metadata, "relationship_type")
            and edge_metadata.relationship_type
        ):
            relationship_key = edge_metadata.relationship_type

        property_key = str(data_point.id) + relationship_key + str(field_value.id)
        visited_properties[property_key] = True

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
