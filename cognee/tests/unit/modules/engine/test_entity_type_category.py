import pytest

from cognee.modules.engine.models import EntityType, EntityTypeCategory


def test_category_defaults_to_unclassified():
    """None must stay distinct from "other", or "no category yet" could not be computed."""
    entity_type = EntityType(name="person", description="person")

    assert entity_type.category is None


def test_category_does_not_change_the_node_id():
    """Identity is the name only, so graphs written before the field keep their ids."""
    plain = EntityType(name="person", description="person")
    categorized = EntityType(name="person", description="person", category="person")

    assert plain.id == categorized.id == EntityType.id_for("person")


def test_node_stored_before_the_field_existed_still_loads():
    stored_props = {"name": "country", "description": "country"}

    assert EntityType(**stored_props).category is None


def test_a_value_this_taxonomy_lacks_still_loads():
    """A reader that rebuilds EntityType from stored properties must not fail on a
    category a newer version wrote."""
    assert EntityType(name="country", description="country", category="geography").category == (
        "geography"
    )


@pytest.mark.parametrize("category", list(EntityTypeCategory))
def test_enum_member_name_equals_its_value(category):
    """BAML registers an enum by member name and validates the answer by value."""
    assert category.name == category.value
