from datetime import datetime, timezone
from cognee.infrastructure.engine import DataPoint
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
    properties_to_visit = set()

    for field_name, field_value in data_point:
        if field_name == "metadata":
            continue

        if isinstance(field_value, DataPoint):
            excluded_properties.add(field_name)

            property_key = str(data_point.id) + field_name + str(field_value.id)

            if property_key in visited_properties:
                continue

            properties_to_visit.add(field_name)

            continue

        if (
            isinstance(field_value, list)
            and len(field_value) > 0
            and isinstance(field_value[0], DataPoint)
        ):
            excluded_properties.add(field_name)

            for index, item in enumerate(field_value):
                property_key = str(data_point.id) + field_name + str(item.id)

                if property_key in visited_properties:
                    continue

                properties_to_visit.add(f"{field_name}.{index}")

            continue

        data_point_properties[field_name] = field_value

    if include_root and str(data_point.id) not in added_nodes:
        SimpleDataPointModel = copy_model(
            type(data_point),
            exclude_fields=list(excluded_properties),
        )
        nodes.append(SimpleDataPointModel(**data_point_properties))
        added_nodes[str(data_point.id)] = True

    for field_name in properties_to_visit:
        index = None

        if "." in field_name:
            field_name, index = field_name.split(".")

        field_value = getattr(data_point, field_name)

        if index is not None:
            field_value = field_value[int(index)]

        edge_key = str(data_point.id) + str(field_value.id) + field_name

        if str(edge_key) not in added_edges:
            edges.append(
                (
                    data_point.id,
                    field_value.id,
                    field_name,
                    {
                        "source_node_id": data_point.id,
                        "target_node_id": field_value.id,
                        "relationship_name": field_name,
                        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    },
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
