"""The HTTP endpoints take node_type as a class name; the resolver turns it into the class."""

import json

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.migration.snapshot import datapoint_registry, rehydrate_node
from cognee.modules.search.methods.node_type_resolution import resolve_node_type
from cognee.shared.graph_model_utils import (
    GENERATED_GRAPH_MODELS_MODULE,
    graph_model_to_graph_schema,
    graph_schema_to_graph_model,
)


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


def test_a_snapshot_stand_in_class_does_not_shadow_the_real_one():
    # A snapshot import of a node whose type is not loaded builds a DataPoint subclass
    # of that name and keeps it for the process; the registry it is given here is
    # empty, so one is built for TextDocument even though the real class is loaded.
    stand_in = type(rehydrate_node({"type": "TextDocument", "name": "memory"}, registry={}))
    assert stand_in is not TextDocument

    assert resolve_node_type("TextDocument") is TextDocument


@pytest.mark.parametrize("name", ["Nope", "schemacolumn", ""])
def test_unknown_names_are_a_422_naming_the_value(name):
    with pytest.raises(CogneeValidationError) as error:
        resolve_node_type(name)

    assert error.value.status_code == 422
    assert repr(name) in error.value.message


def test_a_dotted_module_path_is_refused_like_an_unknown_name():
    """The registry also keys ``module.ClassName``; the wire takes the class name alone."""
    dotted = f"{Entity.__module__}.{Entity.__name__}"
    assert datapoint_registry()[dotted] is Entity

    with pytest.raises(CogneeValidationError) as error:
        resolve_node_type(dotted)

    assert error.value.status_code == 422
    assert repr(dotted) in error.value.message


class _Visitor(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class _VisitorGraph(DataPoint):
    visitors: list[_Visitor]


def test_a_class_a_remote_cognify_built_from_a_schema_does_not_resolve():
    """The classes a remote cognify rebuilds from a request's graph_model schema stay
    loaded on the worker that served it, so resolving them would make the answer depend
    on which worker takes the request. The name is refused on every worker instead."""
    # A schema naming a class no loaded module has, the way a remote caller's would.
    schema = json.loads(
        json.dumps(graph_model_to_graph_schema(_VisitorGraph)).replace("_Visitor", "RemoteGuest")
    )
    graph_schema_to_graph_model(schema)
    assert datapoint_registry()["RemoteGuest"].__module__ == GENERATED_GRAPH_MODELS_MODULE

    with pytest.raises(CogneeValidationError) as error:
        resolve_node_type("RemoteGuest")

    assert error.value.status_code == 422
