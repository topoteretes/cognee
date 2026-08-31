import logging

import pytest

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import NodeSet
from cognee.modules.graph.utils import get_graph_from_model
from cognee.modules.graph.utils.unwrap_transparent_nodes import _WARNED_DROPPED_FIELDS


class Car(DataPoint):
    name: str


class Person(DataPoint):
    name: str
    knows: list | None = None
    owns: Edge | None = None
    cars: list[Car] | None = None
    belongs_to_set: list | None = None
    friend: "Person | None" = None


Person.model_rebuild()


class SocialGraph(DataPoint):
    friends_with: list[Edge[Person, Person]] = []


class Crowd(DataPoint):
    friends_with: list[Edge[Person, Person]] = []
    staffed_by: tuple | None = None
    members: list[Person] | None = None
    metadata: dict = {"index_fields": [], "transparent": True}


def _rel_names(edges):
    return {edge[2] for edge in edges}


def _ids(nodes):
    return {str(node.id) for node in nodes}


@pytest.mark.asyncio
async def test_two_relationship_types_to_the_same_target_both_survive():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    alice.knows = [
        (Edge(relationship_type="friends_with"), bob),
        (Edge(relationship_type="married_to"), bob),
    ]

    _, edges = await get_graph_from_model(alice)
    assert {"friends_with", "married_to"}.issubset(_rel_names(edges))
    matching = [e for e in edges if e[2] in {"friends_with", "married_to"}]
    assert len(matching) == 2


@pytest.mark.asyncio
async def test_explicit_edge_on_container_emits_alice_to_bob():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    graph = SocialGraph(friends_with=[Edge(source=alice, target=bob)])

    nodes, edges = await get_graph_from_model(graph)
    stored = next(node for node in nodes if str(node.id) == str(graph.id))
    assert not hasattr(stored, "friends_with")
    assert any(
        str(e[0]) == str(alice.id) and str(e[1]) == str(bob.id) and e[2] == "friends_with"
        for e in edges
    )


@pytest.mark.asyncio
async def test_endpoints_only_inside_an_edge_become_nodes_without_a_self_edge():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    graph = SocialGraph(friends_with=[Edge(source=alice, target=bob)])

    nodes, edges = await get_graph_from_model(graph)
    assert {str(alice.id), str(bob.id)}.issubset(_ids(nodes))
    assert not any(str(e[0]) == str(alice.id) and str(e[1]) == str(alice.id) for e in edges)


@pytest.mark.asyncio
async def test_local_edge_stores_weight_not_endpoints():
    car = Car(name="Beetle")
    alice = Person(name="Alice", owns=Edge(target=car, weight=0.8))

    _, edges = await get_graph_from_model(alice)
    owns = next(e for e in edges if e[2] == "owns")
    assert str(owns[0]) == str(alice.id)
    assert str(owns[1]) == str(car.id)
    assert owns[3]["weight"] == 0.8
    assert "source" not in owns[3]
    assert "target" not in owns[3]
    assert "relationship_type" not in owns[3]


@pytest.mark.asyncio
async def test_nested_list_edge_properties_are_the_four_standard_keys():
    car = Car(name="Beetle")
    alice = Person(name="Alice", cars=[car])

    _, edges = await get_graph_from_model(alice)
    props = next(e for e in edges if e[2] == "cars")[3]
    assert set(props) == {"source_node_id", "target_node_id", "relationship_name", "updated_at"}


@pytest.mark.asyncio
async def test_belongs_to_set_is_stored_as_names_and_walked():
    node_set = NodeSet(name="team")
    alice = Person(name="Alice", belongs_to_set=[node_set])

    nodes, edges = await get_graph_from_model(alice)
    stored = next(node for node in nodes if str(node.id) == str(alice.id))
    assert stored.belongs_to_set == ["team"]
    assert any(e[2] == "belongs_to_set" and str(e[1]) == str(node_set.id) for e in edges)


@pytest.mark.asyncio
async def test_transparent_root_contributes_children_not_itself():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    crowd = Crowd(members=[alice, bob])

    nodes, _ = await get_graph_from_model(crowd)
    assert str(crowd.id) not in _ids(nodes)
    assert {str(alice.id), str(bob.id)}.issubset(_ids(nodes))


@pytest.mark.asyncio
async def test_transparent_foreign_source_edge_is_skipped_whole(caplog):
    _WARNED_DROPPED_FIELDS.clear()
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    crowd = Crowd(friends_with=[Edge(source=alice, target=bob)])

    with caplog.at_level(logging.WARNING):
        nodes, edges = await get_graph_from_model(crowd)

    assert nodes == []
    assert edges == []
    warnings = [r for r in caplog.records if "half-hoisted" in r.getMessage()]
    assert len(warnings) == 1

    _WARNED_DROPPED_FIELDS.clear()
    child = Person(name="Cara")
    crowd = Crowd(
        friends_with=[Edge(source=alice, target=bob)],
        staffed_by=(Edge(relationship_type="staffed_by"), child),
    )
    nodes, _edges = await get_graph_from_model(crowd)
    assert str(child.id) in _ids(nodes)
    assert str(alice.id) not in _ids(nodes)
    assert str(bob.id) not in _ids(nodes)


@pytest.mark.asyncio
async def test_self_reference_terminates():
    alice = Person(name="Alice")
    alice.friend = alice
    nodes, edges = await get_graph_from_model(alice)
    assert str(alice.id) in _ids(nodes)
    assert any(e[2] == "friend" for e in edges)
