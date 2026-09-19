from typing import Literal

import pytest
from pydantic import ValidationError

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge


class Person(DataPoint):
    name: str


class Company(DataPoint):
    name: str


class Car(DataPoint):
    name: str


class SocialGraph(DataPoint):
    friends: list[Edge[Person, Person]] = []


def test_bare_edge_constructs_and_dumps_weight():
    edge = Edge(weight=0.8)
    assert edge.to_properties() == {"weight": 0.8}


def test_to_properties_keeps_relationship_type_for_stored_edge_readers():
    # Redundant with the first-class relationship_name, kept because stored-edge
    # readers still look it up in the property bag; migration tracked separately.
    edge = Edge(relationship_type="purchased", weight=0.8)
    assert edge.to_properties() == {"relationship_type": "purchased", "weight": 0.8}


def test_two_type_arguments_resolve_defaults():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    edge = Edge[Person, Person](source=alice, target=bob)
    assert edge.source is alice
    assert edge.target is bob


def test_wrong_endpoint_type_raises():
    alice = Person(name="Alice")
    acme = Company(name="Acme")
    with pytest.raises(ValidationError):
        Edge[Person, Company](source=acme, target=alice)


def test_literal_relationship_type_rejects_unknown_name():
    with pytest.raises(ValidationError):
        Edge[Person, Person, Literal["a", "b"]](relationship_type="c")


def test_bare_edge_assigned_into_parametrized_list_keeps_source_identity():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    graph = SocialGraph(friends=[Edge(source=alice, target=bob)])
    assert graph.friends[0].source is alice


def test_fill_endpoints_fills_source_by_identity_and_does_not_mutate():
    alice = Person(name="Alice")
    car = Car(name="Beetle")
    original = Edge(target=car, weight=0.8)

    filled = original.fill_endpoints(alice, "owns")

    assert filled.source is alice
    assert filled.target is car
    assert filled.relationship_type == "owns"
    assert original.source is None


def test_fill_endpoints_does_not_rewrite_relationship_type():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    filled = Edge(relationship_type="Mentors On").fill_endpoints(alice, "x", target=bob)
    assert filled.relationship_type == "Mentors On"


def test_fill_endpoints_returns_self_when_already_complete():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    edge = Edge(source=alice, target=bob, relationship_type="friends_with")
    assert edge.fill_endpoints(alice, "friends_with") is edge


def test_fill_endpoints_target_argument_wins():
    alice = Person(name="Alice")
    car = Car(name="Beetle")
    bob = Person(name="Bob")
    filled = Edge(target=car).fill_endpoints(alice, "owns", target=bob)
    assert filled.target is bob


def test_fill_endpoints_without_target_raises():
    alice = Person(name="Alice")
    with pytest.raises(ValueError):
        Edge().fill_endpoints(alice, "owns")


def test_to_properties_excludes_endpoints():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    props = Edge(source=alice, target=bob, weight=0.5).to_properties()
    assert "source" not in props
    assert "target" not in props
    assert props == {"weight": 0.5}


def test_model_construct_to_properties_keeps_only_the_name():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    edge = Edge.model_construct(source=alice, target=bob, relationship_type="x")
    assert edge.to_properties() == {"relationship_type": "x"}


def test_fill_endpoints_source_fallback_rejects_a_container_owner():
    """An omitted source on a parametrized edge must not make the container the subject."""
    bob = Person(name="Bob")
    graph = SocialGraph(friends=[Edge(target=bob)])

    with pytest.raises(ValueError, match="friends"):
        graph.friends[0].fill_endpoints(graph, "friends")


def test_fill_endpoints_source_fallback_accepts_a_declared_source_owner():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    edge = Edge[Person, Person].model_validate(Edge(target=bob))

    filled = edge.fill_endpoints(alice, "friends")

    assert filled.source is alice


def test_fill_endpoints_source_fallback_accepts_string_declared_owner():
    """Self-referential edges name their endpoints as strings; match by class name."""

    class Colleague(DataPoint):
        name: str
        works_with: list[Edge["Colleague", "Colleague"]] = []

    one = Colleague(name="One")
    other = Colleague(name="Other")
    one.works_with = [Edge(target=other)]

    filled = one.works_with[0].fill_endpoints(one, "works_with")

    assert filled.source is one


def test_fill_endpoints_source_fallback_accepts_a_plain_model_copy_owner():
    """copy_model copies keep the class name but not the class; match by MRO names."""
    from cognee.modules.storage.utils import copy_model

    PersonCopy = copy_model(Person)
    owner = PersonCopy(name="Alice")
    bob = Person(name="Bob")
    edge = Edge[Person, Person].model_validate(Edge(target=bob))

    filled = edge.fill_endpoints(owner, "friends")

    assert filled.source is owner


def test_fill_endpoints_unparametrized_edge_keeps_the_permissive_fallback():
    """The tuple form and other legacy spellings record no generics; owner always fits."""

    class Container(DataPoint):
        name: str

    container = Container(name="box")
    bob = Person(name="Bob")

    filled = Edge(target=bob).fill_endpoints(container, "links")

    assert filled.source is container
