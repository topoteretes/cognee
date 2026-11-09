"""The HTTP endpoints take node_type as a class name; the resolver turns it into the class."""

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.search.methods.node_type_resolution import resolve_node_type


def test_omitted_name_is_the_node_set_default():
    assert resolve_node_type(None) is NodeSet


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("NodeSet", "cognee.modules.engine.models.node_set"),
        ("Entity", "cognee.modules.engine.models.Entity"),
        ("TextDocument", "cognee.modules.data.processing.document_types.TextDocument"),
        ("SchemaColumn", "cognee.tasks.schema.models"),
    ],
)
def test_known_names_resolve_to_the_real_class(name, module):
    node_type = resolve_node_type(name)

    assert node_type.__name__ == name
    assert node_type.__module__ == module


def test_entity_resolves_to_the_class_the_scope_rules_compare_against():
    assert resolve_node_type("Entity") is Entity


@pytest.mark.parametrize("name", ["Nope", "schemacolumn", ""])
def test_unknown_names_are_a_422_naming_the_value(name):
    with pytest.raises(CogneeValidationError) as error:
        resolve_node_type(name)

    assert error.value.status_code == 422
    assert repr(name) in error.value.message
