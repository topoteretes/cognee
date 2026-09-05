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


def test_to_properties_excludes_relationship_type():
    edge = Edge(relationship_type="purchased", weight=0.8)
    assert edge.to_properties() == {"weight": 0.8}


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


def test_normalize_fills_source_by_identity_and_does_not_mutate():
    alice = Person(name="Alice")
    car = Car(name="Beetle")
    original = Edge(target=car, weight=0.8)

    filled = original.normalize(alice, "owns")

    assert filled.source is alice
    assert filled.target is car
    assert filled.relationship_type == "owns"
    assert original.source is None


def test_normalize_does_not_rewrite_relationship_type():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    filled = Edge(relationship_type="Mentors On").normalize(alice, "x", target=bob)
    assert filled.relationship_type == "Mentors On"


def test_normalize_returns_self_when_already_complete():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    edge = Edge(source=alice, target=bob, relationship_type="friends_with")
    assert edge.normalize(alice, "friends_with") is edge


def test_normalize_target_argument_wins():
    alice = Person(name="Alice")
    car = Car(name="Beetle")
    bob = Person(name="Bob")
    filled = Edge(target=car).normalize(alice, "owns", target=bob)
    assert filled.target is bob


def test_normalize_without_target_raises():
    alice = Person(name="Alice")
    with pytest.raises(ValueError):
        Edge().normalize(alice, "owns")


def test_to_properties_excludes_endpoints():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    props = Edge(source=alice, target=bob, weight=0.5).to_properties()
    assert "source" not in props
    assert "target" not in props
    assert props == {"weight": 0.5}


def test_model_construct_to_properties_is_empty():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    edge = Edge.model_construct(source=alice, target=bob, relationship_type="x")
    assert edge.to_properties() == {}
