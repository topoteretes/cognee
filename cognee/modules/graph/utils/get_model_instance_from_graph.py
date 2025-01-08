from pydantic_core import PydanticUndefined
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


def get_model_instance_from_graph(nodes: list[DataPoint], edges: list, entity_id: str):
    node_map = {}

    for node in nodes:
        node_map[node.id] = node

    for edge in edges:
        source_node = node_map[edge[0]]
        target_node = node_map[edge[1]]
        edge_label = edge[2]
        edge_properties = edge[3] if len(edge) == 4 else {}
        edge_metadata = edge_properties.get("metadata", {})
        edge_type = edge_metadata.get("type")

        if edge_type == "list":
            NewModel = copy_model(
                type(source_node), {edge_label: (list[type(target_node)], PydanticUndefined)}
            )

            node_map[edge[0]] = NewModel(**source_node.model_dump(), **{edge_label: [target_node]})
        else:
            NewModel = copy_model(
                type(source_node), {edge_label: (type(target_node), PydanticUndefined)}
            )

            node_map[edge[0]] = NewModel(**source_node.model_dump(), **{edge_label: target_node})

    return node_map[entity_id]
