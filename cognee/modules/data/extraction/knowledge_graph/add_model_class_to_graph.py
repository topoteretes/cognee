# PROPOSED TO BE DEPRECATED

from typing import Type, Optional, get_args, get_origin
from pydantic import BaseModel
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


async def add_model_class_to_graph(
    model_class: Type[BaseModel],
    graph: GraphDBInterface,
    parent: Optional[str] = None,
    relationship: Optional[str] = None,
):
    model_name = model_class.__name__

    if await graph.extract_node(model_name):
        return

    await graph.add_node(model_name, dict(type="model"))

    if parent and relationship:
        await graph.add_edge(
            parent,
            model_name,
            relationship,
            dict(
                relationship_name=relationship,
                source_node_id=parent,
                target_node_id=model_name,
            ),
        )

    for field_name, field in model_class.model_fields.items():
        original_types = get_args(field.annotation)
        field_type = original_types[0] if len(original_types) > 0 else None

        if field_type is None:
            continue

        if hasattr(field_type, "model_fields"):  # Check if field type is a Pydantic model
            await add_model_class_to_graph(field_type, graph, model_name, field_name)
        elif isinstance(get_origin(field.annotation), list):
            list_types = get_args(field_type)
            for item_type in list_types:
                await add_model_class_to_graph(item_type, graph, model_name, field_name)
        elif isinstance(field_type, list):
            item_type = get_args(field_type)[0]
            if hasattr(item_type, "model_fields"):
                await add_model_class_to_graph(item_type, graph, model_name, field_name)
            else:
                await graph.add_node(str(item_type), dict(type="value"))
                await graph.add_edge(
                    model_name,
                    str(item_type),
                    field_name,
                    dict(
                        relationship_name=field_name,
                        source_node_id=model_name,
                        target_node_id=str(item_type),
                    ),
                )
        else:
            await graph.add_node(str(field_type), dict(type="value"))
            await graph.add_edge(
                model_name,
                str(field_type),
                field_name,
                dict(
                    relationship_name=field_name,
                    source_node_id=model_name,
                    target_node_id=str(field_type),
                ),
            )
