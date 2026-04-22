from pydantic_core import PydanticUndefined
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


# Memoize extended-model classes across calls. ``copy_model`` returns a
# brand-new pydantic subclass on every invocation, and each one attaches
# per-class validator/serializer state to pydantic's global caches that's
# never released. Keying by ``(source type, target type, edge label, is_list)``
# means a single class per unique relationship shape.
_EXTENDED_MODEL_CACHE: dict = {}


def _extended_model_for(source_type, target_type, edge_label, is_list: bool):
    key = (source_type, target_type, edge_label, is_list)
    cached = _EXTENDED_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    annotation = list[target_type] if is_list else target_type
    model = copy_model(source_type, {edge_label: (annotation, PydanticUndefined)})
    _EXTENDED_MODEL_CACHE[key] = model
    return model


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

        is_list = edge_type == "list"
        NewModel = _extended_model_for(
            type(source_node), type(target_node), edge_label, is_list
        )

        if is_list:
            # Preserve any targets already attached to this (source, edge)
            # from an earlier edge in the loop — otherwise multi-target list
            # relationships lose all but the last iteration's target.
            existing = getattr(source_node, edge_label, None)
            if isinstance(existing, list):
                value = existing + [target_node]
            else:
                value = [target_node]
        else:
            value = target_node

        dump = source_node.model_dump()
        # Drop the field we're about to overwrite so the kwargs form isn't
        # a duplicate keyword, and so a previously-list value on the dumped
        # dict doesn't collide with the new list.
        dump.pop(edge_label, None)
        node_map[edge[0]] = NewModel(**dump, **{edge_label: value})

    return node_map[entity_id]
