from datetime import datetime, timezone

from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model

async def get_graph_from_model(
    data_point: DataPoint,
    include_root = True,
    added_nodes = None,
    added_edges = None,
):
    if data_point.id in added_nodes:
        return [], []

    nodes = []
    edges = []
    added_nodes = added_nodes or {}
    added_edges = added_edges or {}

    data_point_properties = {}
    excluded_properties = set()

    for field_name, field_value in data_point:
        if field_name == "_metadata":
            continue

        if field_value is None:
            excluded_properties.add(field_name)
            continue

        if isinstance(field_value, DataPoint):
            excluded_properties.add(field_name)
            nodes, edges, added_nodes, added_edges = add_nodes_and_edges(
                data_point,
                field_name,
                field_value,
                nodes,
                edges,
                added_nodes,
                added_edges,
            )

            property_nodes, property_edges = await get_graph_from_model(
                field_value,
                True,
                added_nodes,
                added_edges,
            )

            for node in property_nodes:
                if str(node.id) not in added_nodes:
                    nodes.append(node)
                    added_nodes[str(node.id)] = True

            for edge in property_edges:
                edge_key = str(edge[0]) + str(edge[1]) + edge[2]

                if str(edge_key) not in added_edges:
                    edges.append(edge)
                    added_edges[str(edge_key)] = True

            for property_node in get_own_properties(property_nodes, property_edges):
                edge_key = str(data_point.id) + str(property_node.id) + field_name

                if str(edge_key) not in added_edges:
                    edges.append((data_point.id, property_node.id, field_name, {
                        "source_node_id": data_point.id,
                        "target_node_id": property_node.id,
                        "relationship_name": field_name,
                        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    }))
                    added_edges[str(edge_key)] = True
            continue

        if isinstance(field_value, list) and len(field_value) > 0 and isinstance(field_value[0], DataPoint):
            excluded_properties.add(field_name)

            for item in field_value:
                property_nodes, property_edges = await get_graph_from_model(
                    item,
                    True,
                    added_nodes,
                    added_edges,
                )

                for node in property_nodes:
                    if str(node.id) not in added_nodes:
                        nodes.append(node)
                        added_nodes[str(node.id)] = True

                for edge in property_edges:
                    edge_key = str(edge[0]) + str(edge[1]) + edge[2]

                    if str(edge_key) not in added_edges:
                        edges.append(edge)
                        added_edges[edge_key] = True

                for property_node in get_own_properties(property_nodes, property_edges):
                    edge_key = str(data_point.id) + str(property_node.id) + field_name

                    if str(edge_key) not in added_edges:
                        edges.append((data_point.id, property_node.id, field_name, {
                            "source_node_id": data_point.id,
                            "target_node_id": property_node.id,
                            "relationship_name": field_name,
                            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "metadata": {
                                "type": "list"
                            },
                        }))
                        added_edges[edge_key] = True
            continue

        data_point_properties[field_name] = field_value

    if include_root:
        SimpleDataPointModel = copy_model(
            type(data_point),
            include_fields = {
                "_metadata": (dict, data_point._metadata),
                "__tablename__": data_point.__tablename__,
            },
            exclude_fields = excluded_properties,
        )
        nodes.append(SimpleDataPointModel(**data_point_properties))
        added_nodes[str(data_point.id)] = True

    return nodes, edges


def add_nodes_and_edges(
    data_point, field_name, field_value, nodes, edges, added_nodes, added_edges
):

    property_nodes, property_edges = get_graph_from_model(
        field_value, dict(added_nodes), dict(added_edges)
    )

    for node in property_nodes:
        if str(node.id) not in added_nodes:
            nodes.append(node)
            added_nodes[str(node.id)] = True

    for edge in property_edges:
        edge_key = str(edge[0]) + str(edge[1]) + edge[2]

        if str(edge_key) not in added_edges:
            edges.append(edge)
            added_edges[str(edge_key)] = True

    for property_node in get_own_properties(property_nodes, property_edges):
        edge_key = str(data_point.id) + str(property_node.id) + field_name

        if str(edge_key) not in added_edges:
            edges.append(
                (
                    data_point.id,
                    property_node.id,
                    field_name,
                    {
                        "source_node_id": data_point.id,
                        "target_node_id": property_node.id,
                        "relationship_name": field_name,
                        "updated_at": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    },
                )
            )
            added_edges[str(edge_key)] = True

    return (nodes, edges, added_nodes, added_edges)


def get_own_properties(property_nodes, property_edges):
    own_properties = []

    destination_nodes = [str(property_edge[1]) for property_edge in property_edges]

    for node in property_nodes:
        if str(node.id) in destination_nodes:
            continue

        own_properties.append(node)

    return own_properties
