import logging
from typing import Annotated, Literal

import pytest
from pydantic import ValidationError

from cognee.infrastructure.engine import DataPoint, Edge, FromIdentity
from cognee.modules.graph.utils import get_graph_from_model
from cognee.shared.graph_model_utils import (
    content_graph_to_data_point,
    datapoint_model_to_basemodel,
)


class Role(DataPoint):
    name: str
    rank: str = "member"
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    is_a: Annotated[Role, FromIdentity()]
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class NamedPerson(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Student(NamedPerson):
    pass


class PeopleGraph(DataPoint):
    people: list[Person]


class MixedGraph(DataPoint):
    people: list[Person]
    friends_with: list[Edge[Person, Person]]


class NestedOnly(DataPoint):
    name: str
    child: "NestedOnly | None" = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


NestedOnly.model_rebuild()


class FixedGraph(DataPoint):
    people: list[NamedPerson]
    friends_with: list[Edge[NamedPerson, NamedPerson]]


class EnumeratedGraph(DataPoint):
    people: list[NamedPerson]
    ties: list[Edge[NamedPerson, NamedPerson, Literal["friends_with", "married_to", "sibling_of"]]]


class FreeFormGraph(DataPoint):
    people: list[NamedPerson]
    ties: list[Edge[NamedPerson, NamedPerson, str]]


class Group(DataPoint):
    name: str
    members: list[NamedPerson]


class GroupsGraph(DataPoint):
    groups: list[Group]
    friends_with: list[Edge[NamedPerson, NamedPerson]]


class Clique(DataPoint):
    label: str
    members: list[NamedPerson]
    friends_with: list[Edge[NamedPerson, NamedPerson]] = []


class Crowd(DataPoint):
    cliques: list[Clique]


class Campus(DataPoint):
    people: list[NamedPerson]
    students: list[Student]
    friends_with: list[Edge[NamedPerson, NamedPerson]]


class Dump:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self):
        return self._payload


async def _from_dump(model: type[DataPoint], dump: dict):
    simplified = datapoint_model_to_basemodel(model, strip_metadata=True)
    content_graph = simplified.model_validate(dump)
    return await content_graph_to_data_point(content_graph, model)


def _rel_names(edges):
    return {edge[2] for edge in edges}


@pytest.mark.asyncio
async def test_from_identity_strings_become_nodes_deduped_by_id():
    simplified = datapoint_model_to_basemodel(PeopleGraph, strip_metadata=True)
    content_graph = simplified(
        people=[
            {"name": "Alice", "is_a": "Student"},
            {"name": "Bob", "is_a": "Student"},
        ]
    )

    root = await content_graph_to_data_point(content_graph, PeopleGraph)
    assert root.people[0].is_a.id == Role.id_for("Student")
    assert root.people[1].is_a.id == Role.id_for("Student")

    nodes, _ = await get_graph_from_model(root)
    role_nodes = [node for node in nodes if getattr(node, "name", None) == "Student"]
    assert len(role_nodes) == 1


@pytest.mark.asyncio
async def test_from_identity_keeps_defaulted_extra_fields():
    simplified = datapoint_model_to_basemodel(Person, strip_metadata=True)
    content_graph = simplified(name="Alice", is_a="Student")
    person = await content_graph_to_data_point(content_graph, Person)
    assert person.is_a.rank == "member"


@pytest.mark.asyncio
async def test_model_without_from_identity_matches_model_validate():
    simplified = datapoint_model_to_basemodel(NestedOnly, strip_metadata=True)
    content_graph = simplified(name="root", child={"name": "leaf"})
    dump = content_graph.model_dump()
    via_inverse = await content_graph_to_data_point(content_graph, NestedOnly)
    via_validate = NestedOnly.model_validate(dump)
    assert via_inverse.name == via_validate.name
    assert via_inverse.child.name == via_validate.child.name


@pytest.mark.asyncio
async def test_fixed_row_names_edge_after_the_field():
    root = await _from_dump(
        FixedGraph,
        {
            "people": [{"name": "Alice"}, {"name": "Bob"}],
            "friends_with": [{"source": "Alice", "target": "Bob"}],
        },
    )
    assert root.friends_with[0].relationship_type == "friends_with"
    assert root.friends_with[0].source.id == NamedPerson.id_for("Alice")
    assert root.friends_with[0].source is not root


@pytest.mark.asyncio
async def test_enumerated_row_keeps_relationship_type():
    root = await _from_dump(
        EnumeratedGraph,
        {
            "people": [{"name": "Alice"}, {"name": "Bob"}],
            "ties": [{"source": "Alice", "target": "Bob", "relationship_type": "married_to"}],
        },
    )
    assert root.ties[0].relationship_type == "married_to"


@pytest.mark.asyncio
async def test_free_form_row_normalizes_name():
    root = await _from_dump(
        FreeFormGraph,
        {
            "people": [{"name": "Alice"}, {"name": "Bob"}],
            "ties": [{"source": "Alice", "target": "Bob", "relationship_type": "Mentors On"}],
        },
    )
    assert root.ties[0].relationship_type == "mentors_on"


@pytest.mark.asyncio
async def test_edge_field_without_default_validates():
    root = await content_graph_to_data_point(Dump({"people": [{"name": "Alice"}]}), FixedGraph)
    assert root.friends_with == []


@pytest.mark.asyncio
async def test_same_entity_under_two_parents_resolves():
    root = await _from_dump(
        GroupsGraph,
        {
            "groups": [
                {"name": "g1", "members": [{"name": "Alice"}]},
                {"name": "g2", "members": [{"name": "Alice"}, {"name": "Bob"}]},
            ],
            "friends_with": [{"source": "Alice", "target": "Bob"}],
        },
    )
    alice_ids = {
        member.id for group in root.groups for member in group.members if member.name == "Alice"
    }
    assert alice_ids == {NamedPerson.id_for("Alice")}
    assert root.friends_with[0].source.id == NamedPerson.id_for("Alice")
    assert root.friends_with[0].target.id == NamedPerson.id_for("Bob")


@pytest.mark.asyncio
async def test_unknown_endpoint_is_skipped(caplog):
    with caplog.at_level(logging.WARNING):
        root = await _from_dump(
            FixedGraph,
            {
                "people": [{"name": "Alice"}, {"name": "Bob"}],
                "friends_with": [
                    {"source": "Alice", "target": "Bob"},
                    {"source": "Alice", "target": "Nobody"},
                ],
            },
        )
    assert len(root.friends_with) == 1
    assert root.friends_with[0].target.id == NamedPerson.id_for("Bob")
    assert "friends_with" in caplog.text


@pytest.mark.asyncio
async def test_subclass_endpoint_is_skipped(caplog):
    with caplog.at_level(logging.WARNING):
        root = await _from_dump(
            Campus,
            {
                "people": [{"name": "Bob"}],
                "students": [{"name": "Alice"}],
                "friends_with": [{"source": "Alice", "target": "Bob"}],
            },
        )
    assert root.friends_with == []
    assert "friends_with" in caplog.text


@pytest.mark.asyncio
async def test_case_and_spacing_in_endpoint_still_resolves():
    root = await _from_dump(
        FixedGraph,
        {
            "people": [{"name": "Alice"}, {"name": "Bob Smith"}],
            "friends_with": [{"source": "ALICE", "target": "bob smith"}],
        },
    )
    assert root.friends_with[0].source.id == NamedPerson.id_for("Alice")
    assert root.friends_with[0].target.id == NamedPerson.id_for("Bob Smith")


@pytest.mark.asyncio
async def test_three_edges_between_the_same_pair_survive():
    root = await _from_dump(
        EnumeratedGraph,
        {
            "people": [{"name": "Alice"}, {"name": "Bob"}],
            "ties": [
                {"source": "Alice", "target": "Bob", "relationship_type": "friends_with"},
                {"source": "Alice", "target": "Bob", "relationship_type": "married_to"},
                {"source": "Alice", "target": "Bob", "relationship_type": "sibling_of"},
            ],
        },
    )
    _, edges = await get_graph_from_model(root)
    assert {"friends_with", "married_to", "sibling_of"}.issubset(_rel_names(edges))
    matching = [edge for edge in edges if edge[2] in {"friends_with", "married_to", "sibling_of"}]
    assert len(matching) == 3


@pytest.mark.asyncio
async def test_enumerated_out_of_literal_raises():
    with pytest.raises(ValidationError):
        await content_graph_to_data_point(
            Dump(
                {
                    "people": [{"name": "Alice"}, {"name": "Bob"}],
                    "ties": [
                        {
                            "source": "Alice",
                            "target": "Bob",
                            "relationship_type": "not_a_friend",
                        }
                    ],
                }
            ),
            EnumeratedGraph,
        )


@pytest.mark.asyncio
async def test_nested_owners_keep_their_own_rows():
    root = await _from_dump(
        Crowd,
        {
            "cliques": [
                {
                    "label": "one",
                    "members": [{"name": "Alice"}, {"name": "Bob"}],
                    "friends_with": [{"source": "Alice", "target": "Bob"}],
                },
                {
                    "label": "two",
                    "members": [{"name": "Carol"}, {"name": "Dave"}],
                    "friends_with": [{"source": "Carol", "target": "Dave"}],
                },
            ]
        },
    )
    assert root.cliques[0].friends_with[0].target.id == NamedPerson.id_for("Bob")
    assert root.cliques[1].friends_with[0].target.id == NamedPerson.id_for("Dave")
    assert len(root.cliques[0].friends_with) == 1
    assert len(root.cliques[1].friends_with) == 1


@pytest.mark.asyncio
async def test_from_identity_and_edge_rows_together():
    root = await _from_dump(
        MixedGraph,
        {
            "people": [
                {"name": "Alice", "is_a": "Student"},
                {"name": "Bob", "is_a": "Student"},
            ],
            "friends_with": [{"source": "Alice", "target": "Bob"}],
        },
    )
    assert root.people[0].is_a.id == Role.id_for("Student")
    assert root.friends_with[0].source.id == Person.id_for("Alice")
    assert root.friends_with[0].source is not root
    _, edges = await get_graph_from_model(root)
    assert "friends_with" in _rel_names(edges)
