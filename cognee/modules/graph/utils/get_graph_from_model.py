from datetime import datetime, timezone
from cognee.infrastructure.engine import DataPoint
from cognee.modules import data
from cognee.modules.storage.utils import copy_model

def get_graph_from_model(data_point: DataPoint, include_root = True):
    nodes = []
    edges = []

    data_point_properties = {}
    excluded_properties = set()

    for field_name, field_value in data_point:
        if field_name == "_metadata":
            continue

        if isinstance(field_value, DataPoint):
            excluded_properties.add(field_name)

            property_nodes, property_edges = get_graph_from_model(field_value, True)
            nodes[:0] = property_nodes
            edges[:0] = property_edges

            for property_node in get_own_properties(property_nodes, property_edges):
                edges.append((data_point.id, property_node.id, field_name, {
                    "source_node_id": data_point.id,
                    "target_node_id": property_node.id,
                    "relationship_name": field_name,
                    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                }))
            continue

        if isinstance(field_value, list):
            if isinstance(field_value[0], DataPoint):
                excluded_properties.add(field_name)

                for item in field_value:
                    property_nodes, property_edges = get_graph_from_model(item, True)
                    nodes[:0] = property_nodes
                    edges[:0] = property_edges

                    for property_node in get_own_properties(property_nodes, property_edges):
                        edges.append((data_point.id, property_node.id, field_name, {
                            "source_node_id": data_point.id,
                            "target_node_id": property_node.id,
                            "relationship_name": field_name,
                            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "metadata": {
                                "type": "list"
                            },
                        }))
                continue

        data_point_properties[field_name] = field_value

    SimpleDataPointModel = copy_model(
        type(data_point),
        include_fields = {
            "_metadata": (dict, data_point._metadata),
        },
        exclude_fields = excluded_properties,
    )

    if include_root:
        nodes.append(SimpleDataPointModel(**data_point_properties))

    return nodes, edges


def get_own_properties(property_nodes, property_edges):
    own_properties = []

    destination_nodes = [str(property_edge[1]) for property_edge in property_edges]

    for node in property_nodes:
        if str(node.id) in destination_nodes:
            continue

        own_properties.append(node)

    return own_properties
