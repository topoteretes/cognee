import json

import pytest
from pydantic import ValidationError

from cognee.modules.engine.models import EntityType, EntityTypeCategory
from cognee.modules.graph.utils import get_graph_from_model
from cognee.modules.storage.utils import JSONEncoder


def test_category_defaults_to_unclassified():
    """None must stay distinct from "other", or "no category yet" could not be computed."""
    entity_type = EntityType(name="person", description="person")

    assert entity_type.category is None


def test_category_does_not_change_the_node_id():
    """Identity is the name only, so graphs written before the field keep their ids."""
    plain = EntityType(name="person", description="person")
    categorized = EntityType(name="person", description="person", category="person")

    assert plain.id == categorized.id == EntityType.id_for("person")


def test_category_outside_the_taxonomy_is_rejected():
    """An earlier `str | None` field stored "persno" and left the enum unused."""
    with pytest.raises(ValidationError):
        EntityType(name="person", description="person", category="persno")


def test_assigning_a_category_outside_the_taxonomy_is_rejected():
    """The classifier assigns the field on instances it did not construct."""
    entity_type = EntityType(name="person", description="person")

    with pytest.raises(ValidationError):
        entity_type.category = "persno"

    entity_type.category = "person"
    assert entity_type.category == "person"


def test_node_stored_before_the_field_existed_still_loads():
    stored_props = {"name": "country", "description": "country"}

    assert EntityType(**stored_props).category is None


@pytest.mark.asyncio
async def test_the_node_an_adapter_receives_reads_as_the_stored_value():
    """get_graph_from_model rebuilds the node on a plain base class, so the model's own
    config is gone by the time an adapter dumps it. The value must still stringify and
    serialize as "place", not as "EntityTypeCategory.place"."""
    entity_type = EntityType(name="country", description="country", category="place")

    nodes, _ = await get_graph_from_model(entity_type)

    node = next(node for node in nodes if node.id == entity_type.id)
    assert str(node.category) == "place"
    assert json.loads(json.dumps(node.model_dump(), cls=JSONEncoder))["category"] == "place"


def test_enum_member_names_equal_their_values():
    """BAML registers an enum by member name and validates the answer by value."""
    assert all(category.name == category.value for category in EntityTypeCategory)
