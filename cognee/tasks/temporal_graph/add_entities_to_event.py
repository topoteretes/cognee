from cognee.modules.engine.models import Event
from cognee.tasks.temporal_graph.models import EventWithEntities
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.utils import generate_node_id, generate_node_name


def add_entities_to_event(event: Event, event_with_entities: EventWithEntities) -> None:
    """
    Adds extracted entities to an Event object by populating its attributes field.

    For each attribute in the provided EventWithEntities, the function ensures that
    the corresponding entity type exists, creates an Entity node with metadata, and
    links it to the event via an Edge representing the relationship. Entities are
    cached by type to avoid duplication.

    Args:
        event (Event): The target Event object to enrich with entities.
        event_with_entities (EventWithEntities): An event model containing extracted
            attributes with entity, type, and relationship metadata.

    Returns:
        None
    """

    if not event_with_entities.attributes:
        return

    # Create entity types cache
    entity_types = {}

    # Process each attribute
    for attribute in event_with_entities.attributes:
        # Get or create entity type
        entity_type = get_or_create_entity_type(entity_types, attribute.entity_type)

        # Create entity
        entity_id = generate_node_id(attribute.entity)
        entity_name = generate_node_name(attribute.entity)
        entity = Entity(
            id=entity_id,
            name=entity_name,
            is_a=entity_type,
            description=f"Entity {attribute.entity} of type {attribute.entity_type}",
            ontology_valid=False,
            belongs_to_set=None,
        )

        # Create edge
        edge = Edge(relationship_type=attribute.relationship)

        # Add to event attributes
        if event.attributes is None:
            event.attributes = []
        event.attributes.append((edge, [entity]))


def get_or_create_entity_type(entity_types: dict, entity_type_name: str) -> EntityType:
    """
    Retrieves an existing EntityType from the cache or creates a new one if it does not exist.

    If the given entity type name is not already in the cache, a new EntityType is generated
    with a unique ID, normalized name, and description, then added to the cache.

    Args:
        entity_types (dict): A cache mapping entity type names to EntityType objects.
        entity_type_name (str): The name of the entity type to retrieve or create.

    Returns:
        EntityType: The existing or newly created EntityType object.
    """
    if entity_type_name not in entity_types:
        type_id = generate_node_id(entity_type_name)
        type_name = generate_node_name(entity_type_name)
        entity_type = EntityType(
            id=type_id,
            name=type_name,
            type=type_name,
            description=f"Type for {entity_type_name}",
            ontology_valid=False,
        )
        entity_types[entity_type_name] = entity_type

    return entity_types[entity_type_name]
