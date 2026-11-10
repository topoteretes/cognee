"""The HTTP endpoints take node_type as a class name; the resolver turns it into the class."""

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.migration.snapshot import rehydrate_node
from cognee.modules.search.methods import node_type_resolution
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


def test_a_snapshot_stand_in_class_does_not_shadow_the_real_one(monkeypatch):
    # A snapshot import of a node whose type is not loaded builds a DataPoint subclass
    # of that name and keeps it for the process; the registry it is given here is
    # empty, so one is built for TextDocument even though the real class is loaded.
    stand_in = type(rehydrate_node({"type": "TextDocument", "name": "memory"}, registry={}))
    assert stand_in is not TextDocument

    # Where the stand-in sits in the subclass walk depends on which module the process
    # imported first, so pin the order that shadows the real class.
    monkeypatch.setattr(
        node_type_resolution, "get_all_subclasses", lambda cls: [stand_in, TextDocument]
    )

    assert resolve_node_type("TextDocument") is TextDocument


@pytest.mark.parametrize("name", ["Nope", "schemacolumn", ""])
def test_unknown_names_are_a_422_naming_the_value(name):
    with pytest.raises(CogneeValidationError) as error:
        resolve_node_type(name)

    assert error.value.status_code == 422
    assert repr(name) in error.value.message
