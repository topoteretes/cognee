from uuid import uuid4

import pytest

from cognee.modules.engine.models import Entity, EntityType


@pytest.mark.parametrize(
    "name", ["SoftwareSystem", "software system", "Software System", " Software  System "]
)
def test_entity_type_name_and_identity_ignore_case_and_spaces(name):
    entity_type = EntityType(name=name, description="A software system")

    assert entity_type.name == "softwaresystem"
    assert entity_type.id == EntityType.id_for("softwaresystem")
    assert EntityType.id_for(name) == entity_type.id


@pytest.mark.parametrize(
    "name", ["software_system", "software-system", "organization", "department"]
)
def test_different_types_and_other_separators_remain_distinct(name):
    entity_type = EntityType(name=name, description=name)

    assert entity_type.name == name
    assert entity_type.id != EntityType.id_for("softwaresystem")


def test_entity_names_and_identity_keep_their_existing_space_rules():
    entity = Entity(name="Software System", description="A named entity")

    assert entity.name == "Software System"
    assert entity.id == Entity.id_for("Software System")
    assert entity.id != Entity.id_for("SoftwareSystem")
    assert EntityType.id_for("SoftwareSystem") != Entity.id_for("SoftwareSystem")


def test_explicit_type_id_is_preserved():
    explicit_id = uuid4()

    entity_type = EntityType(id=explicit_id, name="Software System", description="type")

    assert entity_type.name == "softwaresystem"
    assert entity_type.id == explicit_id
