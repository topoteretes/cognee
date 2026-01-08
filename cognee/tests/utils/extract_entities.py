from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_node_id, generate_node_name
from cognee.shared.data_models import KnowledgeGraph


def extract_entities(graph: KnowledgeGraph, cache: dict = {}):
    entities = []
    entity_types = []

    for node in graph.nodes:
        node_id = generate_node_id(node.id)

        if node_id not in cache:
            entity = Entity(
                id=node_id,
                name=generate_node_name(node.id),
                type=node.type,
                description=node.description,
                ontology_valid=False,
            )
            cache[node_id] = entity
        else:
            entity = cache[node_id]

        entities.append(entity)

        node_type = node.type
        type_node_id = generate_node_id(node_type)
        if type_node_id not in cache:
            type_node_name = generate_node_name(node_type)

            type_node = EntityType(
                id=type_node_id,
                name=type_node_name,
                type=type_node_name,
                description=type_node_name,
                ontology_valid=False,
            )
            cache[type_node_id] = type_node
        else:
            type_node = cache[type_node_id]

        entity_types.append(type_node)

    return entities + entity_types
