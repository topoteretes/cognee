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


def test_an_explicit_null_stays_none_for_the_sdk_scope_rules():
    assert resolve_node_type(None) is None


def test_node_set_resolves_to_the_class_the_scope_rules_compare_against():
    assert resolve_node_type("NodeSet") is NodeSet


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


class _CustomBase(DataPoint):
    name: str


class _CustomLeaf(_CustomBase):
    """A custom model two levels below DataPoint, as a deployment would import it."""


def test_a_loaded_custom_class_resolves_at_any_depth_of_inheritance():
    assert resolve_node_type("_CustomBase") is _CustomBase
    assert resolve_node_type("_CustomLeaf") is _CustomLeaf


class _Visitor(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class _VisitorGraph(DataPoint):
    visitors: list[_Visitor]


def test_a_class_generated_from_a_graph_model_schema_resolves_once_loaded():
    """The graph-model ingestion path builds classes from a schema; once the serving
    process has built one, its node type resolves like any loaded class."""
    # A schema naming a class no loaded module has, the way a remote caller's would.
    schema = json.loads(
        json.dumps(graph_model_to_graph_schema(_VisitorGraph)).replace("_Visitor", "RemoteGuest")
    )
    graph_model = graph_schema_to_graph_model(schema)
    generated = datapoint_registry()["RemoteGuest"]
    assert generated.__module__ == GENERATED_GRAPH_MODELS_MODULE

    assert resolve_node_type("RemoteGuest") is generated
    assert resolve_node_type(graph_model.__name__) is graph_model


def _snapshot_only_name() -> str:
    """A node type only a snapshot stand-in carries: no real class has the name."""
    rehydrate_node({"type": "SnapshotOnlyType", "name": "memory"}, registry={})
    return "SnapshotOnlyType"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("Nope", id="unknown"),
        pytest.param("schemacolumn", id="wrong case"),
        pytest.param("", id="empty"),
        pytest.param(f"{Entity.__module__}.{Entity.__name__}", id="dotted module path"),
        pytest.param(_snapshot_only_name(), id="snapshot stand-in only"),
    ],
)
def test_unresolvable_names_are_a_422_naming_the_value(name):
    with pytest.raises(CogneeValidationError) as error:
        resolve_node_type(name)

    assert error.value.status_code == 422
    assert repr(name) in error.value.message


def test_the_registry_keys_the_dotted_path_the_resolver_refuses():
    """The registry also keys ``module.ClassName``; the wire takes the class name alone."""
    assert datapoint_registry()[f"{Entity.__module__}.{Entity.__name__}"] is Entity
