"""Typed ``Edge`` and ``FromIdentity`` fields survive the JSON schema round trip (SDK-1408).

The schema is what ``POST /remember``, ``POST /cognify`` and ``cognee.serve()`` move
across the wire. Before this the regenerated model held the prompt model's flat shapes
— edge rows as a nested node type, references as strings — so a remote cognify stored
nodes and attributes where the caller had declared edges and references."""

import json
from typing import Annotated, Literal, get_args

import pytest
from pydantic import ValidationError

from cognee.infrastructure.engine import DataPoint, Edge, FromIdentity
from cognee.shared.exceptions import UnresolvedSchemaReferenceError
from cognee.shared.graph_model_utils import (
    EDGE_KEY,
    FROM_IDENTITY_KEY,
    graph_model_to_graph_schema,
    graph_schema_to_graph_model,
)
from cognee.shared.llm_graph_model import (
    _edge_field_types,
    _identity_field_types,
    datapoint_model_to_basemodel,
)


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Role(DataPoint):
    """Only ever referenced: no field nests a Role object."""

    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Worker(DataPoint):
    name: str
    role: Annotated[Role, FromIdentity()] | None = None
    past_roles: Annotated[list[Role], FromIdentity()] = []
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Colleague(DataPoint):
    name: str
    works_with: list[Edge["Colleague", "Colleague"]] = []
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Graph(DataPoint):
    people: list[Person] = []
    workers: list[Worker] = []
    colleagues: list[Colleague] = []
    ties: list[Edge[Person, Person, Literal["friends_with", "married_to"]]] = []
    knows: list[Edge[Person, Worker]] | None = None
    links: list[Edge[Person, Person, str]] = []


@pytest.fixture(scope="module")
def schema() -> dict:
    return graph_model_to_graph_schema(Graph)


@pytest.fixture(scope="module")
def rebuilt(schema):
    return graph_schema_to_graph_model(json.loads(json.dumps(schema)))


def _nested(model, field):
    """The class inside ``list[X]`` / ``list[X] | None`` on a generated model."""
    annotation = model.model_fields[field].annotation
    members = [arg for arg in get_args(annotation) if arg is not type(None)]
    inner = members[0] if members else annotation
    return get_args(inner)[0] if get_args(inner) else inner


def test_the_schema_names_endpoints_and_targets_and_defines_them(schema):
    assert schema["properties"]["ties"][EDGE_KEY] == {"source": "Person", "target": "Person"}
    assert schema["properties"]["knows"][EDGE_KEY] == {"source": "Person", "target": "Worker"}
    worker = schema["$defs"]["Worker"]["properties"]
    assert worker["role"][FROM_IDENTITY_KEY] == "Role"
    assert worker["past_roles"][FROM_IDENTITY_KEY] == "Role"
    colleague = schema["$defs"]["Colleague"]["properties"]
    assert colleague["works_with"][EDGE_KEY] == {"source": "Colleague", "target": "Colleague"}
    # Role is named only through references, so the prompt model never nests it; the
    # schema must still define it or the server has nothing to point the reference at.
    assert schema["$defs"]["Role"]["properties"]["metadata"]["default"] == {
        "index_fields": ["name"],
        "identity_fields": ["name"],
    }
    # The flat shapes stay: a consumer that ignores the keys reads a valid schema.
    assert schema["$defs"]["TiesEdge"]["properties"]["relationship_type"]["enum"] == [
        "friends_with",
        "married_to",
    ]
    assert worker["role"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_the_rebuilt_model_declares_the_same_edges(rebuilt):
    person, worker = _nested(rebuilt, "people"), _nested(rebuilt, "workers")
    edges = _edge_field_types(rebuilt)

    assert edges["ties"] == (person, person, Literal["friends_with", "married_to"])
    assert edges["links"] == (person, person, str)
    source, target, naming = edges["knows"]
    assert (source, target) == (person, worker)
    assert naming == (str | None), "no relationship_type on the row: named after the field"
    assert type(None) in get_args(rebuilt.model_fields["knows"].annotation)

    colleague = _nested(rebuilt, "colleagues")
    assert _edge_field_types(colleague)["works_with"][:2] == (colleague, colleague)


def test_the_rebuilt_model_declares_the_same_references(rebuilt):
    worker, role = _nested(rebuilt, "workers"), _nested(_nested(rebuilt, "workers"), "past_roles")

    assert _identity_field_types(worker) == {"role": role, "past_roles": role}
    assert role.__name__ == "Role"
    assert type(None) in get_args(worker.model_fields["role"].annotation)


def test_the_server_asks_the_llm_for_the_same_model(rebuilt):
    """What the server's extraction sends the LLM must match the local run's."""
    local = datapoint_model_to_basemodel(Graph).model_json_schema()["$defs"]
    remote = datapoint_model_to_basemodel(rebuilt).model_json_schema()["$defs"]

    for row in ("TiesEdge", "KnowsEdge", "LinksEdge", "WorksWithEdge"):
        assert remote[row] == local[row], row
    for field in ("role", "past_roles"):
        assert remote["Worker"]["properties"][field] == local["Worker"]["properties"][field]
    assert "Role" not in remote, "references are identity strings, never nested objects"


def test_a_parent_validates_the_retyped_child(rebuilt):
    """The rebuild order matters: a parent built before its retyped child would still
    validate the child's old shape (a string where a Role now goes)."""
    person, role = _nested(rebuilt, "people"), _nested(_nested(rebuilt, "workers"), "past_roles")
    ann, bob = person(name="Ann"), person(name="Bob")

    graph = rebuilt(
        people=[ann, bob],
        workers=[{"name": "Cy", "role": role(name="Dev")}],
        ties=[{"source": ann, "target": bob, "relationship_type": "married_to"}],
    )

    assert isinstance(graph.workers[0].role, role)
    assert isinstance(graph.ties[0], Edge)
    assert graph.ties[0].relationship_type == "married_to"
    with pytest.raises(ValidationError):
        rebuilt(ties=[{"source": ann, "target": bob, "relationship_type": "enemies"}])


def test_a_grandparent_validates_the_retyped_grandchild():
    """Two levels up from the retyped class: the rebuild must not reuse a stale
    intermediate schema either."""

    class Team(DataPoint):
        name: str
        members: list[Worker] = []
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class Company(DataPoint):
        teams: list[Team] = []

    rebuilt = graph_schema_to_graph_model(graph_model_to_graph_schema(Company))
    worker = _nested(_nested(rebuilt, "teams"), "members")
    role = _nested(worker, "past_roles")

    company = rebuilt(
        teams=[{"name": "Core", "members": [{"name": "Cy", "role": role(name="Dev")}]}]
    )

    assert isinstance(company.teams[0].members[0].role, role)
    with pytest.raises(ValidationError):
        rebuilt(teams=[{"name": "Core", "members": [{"name": "Cy", "role": "Dev"}]}])


def test_an_extension_naming_an_undefined_class_raises(schema):
    broken = json.loads(json.dumps(schema))
    broken["properties"]["ties"][EDGE_KEY]["target"] = "Ghost"

    with pytest.raises(UnresolvedSchemaReferenceError, match="Ghost"):
        graph_schema_to_graph_model(broken)


def test_a_schema_without_the_keys_is_rebuilt_flat(schema):
    """A hand-written schema (the spec compiler's, today) keeps the old behaviour."""
    flat = json.loads(json.dumps(schema))
    for entry in (flat, *flat["$defs"].values()):
        for prop in entry.get("properties", {}).values():
            prop.pop(EDGE_KEY, None)
            prop.pop(FROM_IDENTITY_KEY, None)

    rebuilt = graph_schema_to_graph_model(flat)

    assert _edge_field_types(rebuilt) == {}
    assert _identity_field_types(_nested(rebuilt, "workers")) == {}
