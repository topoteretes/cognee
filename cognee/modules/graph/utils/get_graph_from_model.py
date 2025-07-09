from datetime import datetime, timezone
from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.storage.utils import copy_model


async def get_graph_from_model(
    data_point: DataPoint,
    added_nodes: dict,
    added_edges: dict,
    visited_properties: dict = None,
    include_root=True,
):
    if str(data_point.id) in added_nodes:
        return [], []

    nodes = []
    edges = []
    visited_properties = visited_properties or {}

    data_point_properties = {
        "type": type(data_point).__name__,
    }
    excluded_properties = set()
    properties_to_visit = []  # Changed from set to list

    for field_name, field_value in data_point:
        if field_name == "metadata":
            continue

        # Check if field_value is a tuple with Edge metadata
        edge_metadata = None
        actual_field_value = field_value

        if isinstance(field_value, tuple) and len(field_value) == 2:
            first_element, second_element = field_value
            if isinstance(first_element, Edge):
                edge_metadata = first_element
                actual_field_value = second_element

        if isinstance(actual_field_value, DataPoint):
            excluded_properties.add(field_name)

            property_key = str(data_point.id) + field_name + str(actual_field_value.id)

            if property_key in visited_properties:
                continue

            properties_to_visit.append((field_name, edge_metadata))  # Changed from add to append

            continue

        if (
            isinstance(actual_field_value, list)
            and len(actual_field_value) > 0
            and isinstance(actual_field_value[0], DataPoint)
        ):
            excluded_properties.add(field_name)

            for index, item in enumerate(actual_field_value):
                property_key = str(data_point.id) + field_name + str(item.id)

                if property_key in visited_properties:
                    continue

                properties_to_visit.append(
                    (f"{field_name}.{index}", edge_metadata)
                )  # Changed from add to append

            continue

        # Only add to properties if it's not a tuple with Edge metadata
        if not (
            isinstance(field_value, tuple)
            and len(field_value) == 2
            and isinstance(field_value[0], Edge)
        ):
            data_point_properties[field_name] = field_value

    if include_root and str(data_point.id) not in added_nodes:
        SimpleDataPointModel = copy_model(
            type(data_point),
            exclude_fields=list(excluded_properties),
        )
        nodes.append(SimpleDataPointModel(**data_point_properties))
        added_nodes[str(data_point.id)] = True

    for property_item in properties_to_visit:
        if isinstance(property_item, tuple):
            field_name_with_index, edge_metadata = property_item
        else:
            # Handle legacy case where properties_to_visit contains just field names
            field_name_with_index, edge_metadata = property_item, None

        index = None
        field_name = field_name_with_index

        if "." in field_name_with_index:
            field_name, index = field_name_with_index.split(".")

        field_value = getattr(data_point, field_name)

        # If field_value is a tuple with Edge metadata, extract the actual value
        if (
            isinstance(field_value, tuple)
            and len(field_value) == 2
            and isinstance(field_value[0], Edge)
        ):
            _, field_value = field_value

        if index is not None:
            field_value = field_value[int(index)]

        edge_key = str(data_point.id) + str(field_value.id) + field_name

        if str(edge_key) not in added_edges:
            # Build edge properties
            edge_properties = {
                "source_node_id": data_point.id,
                "target_node_id": field_value.id,
                "relationship_name": field_name,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add edge metadata if present
            if edge_metadata:
                edge_properties.update(edge_metadata.to_dict())

            edges.append(
                (
                    data_point.id,
                    field_value.id,
                    field_name,
                    edge_properties,
                )
            )
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

        property_key = str(data_point.id) + field_name + str(field_value.id)
        visited_properties[property_key] = True

    return nodes, edges


def get_own_property_nodes(property_nodes, property_edges):
    own_properties = []

    destination_nodes = [str(property_edge[1]) for property_edge in property_edges]

    for node in property_nodes:
        if str(node.id) in destination_nodes:
            continue

        own_properties.append(node)

    return own_properties
