from typing import Callable

from pydantic_core import PydanticUndefined

from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


def merge_dicts(dict1: dict, dict2: dict, agg_fn: Callable) -> dict:
    merged_dict = {}
    for key, value in dict1.items():
        if key in dict2:
            merged_dict[key] = agg_fn(value, dict2[key])
        else:
            merged_dict[key] = value

    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = value
    return merged_dict


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
            new_model_dict = merge_dicts(
                source_node.model_dump(),
                {edge_label: [target_node]},
                lambda a, b: a + b,
            )
            node_map[source_node_id] = NewModel(**new_model_dict)
        else:
            NewModel = copy_model(
                type(source_node), {edge_label: (type(target_node), PydanticUndefined)}
            )

            node_map[target_node_id] = NewModel(
                **source_node.model_dump(), **{edge_label: target_node}
            )

    return node_map[entity_id]
