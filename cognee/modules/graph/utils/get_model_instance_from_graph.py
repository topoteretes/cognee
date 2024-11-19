from typing import Callable

from pydantic_core import PydanticUndefined

from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


def get_model_instance_from_graph(
    nodes: list[DataPoint],
    edges: list[tuple[str, str, str, dict[str, str]]],
    entity_id: str,
):
    node_map = {node.id: node for node in nodes}

    for source_node_id, target_node_id, edge_label, edge_properties in edges:
        source_node = node_map[source_node_id]
        target_node = node_map[target_node_id]
        edge_metadata = edge_properties.get("metadata", {})
        edge_type = edge_metadata.get("type", "default")

        if edge_type == "list":
            NewModel = copy_model(
                type(source_node),
                {edge_label: (list[type(target_node)], PydanticUndefined)},
            )
            source_node_dict = source_node.model_dump()
            source_node_edge_label_values = source_node_dict.get(edge_label, [])
            source_node_dict[edge_label] = source_node_edge_label_values + [target_node]

            node_map[source_node_id] = NewModel(**source_node_dict)
        else:
            NewModel = copy_model(
                type(source_node), {edge_label: (type(target_node), PydanticUndefined)}
            )

            node_map[target_node_id] = NewModel(
                **source_node.model_dump(), **{edge_label: target_node}
            )

    return node_map[entity_id]
