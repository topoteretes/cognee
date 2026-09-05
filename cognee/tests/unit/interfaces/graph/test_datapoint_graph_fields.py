import subprocess
import sys

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import NodeSet


class Person(DataPoint):
    name: str


class Car(DataPoint):
    name: str


class Group(DataPoint):
    members: list[Person]
    metadata: dict = {"index_fields": [], "transparent": True}


class Owner(DataPoint):
    name: str
    owns: Car | None = None
    cars: list[Car] | None = None
    purchased: tuple | None = None
    extra: Edge | None = None
    empty_tuple: tuple | None = None
    mentors: tuple | None = None
    belongs_to_set: list | None = None
    groups: list | None = None
    mixed: tuple | None = None


class SocialGraph(DataPoint):
    friends_with: list[Edge[Person, Person]] = []


class Plain(DataPoint):
    name: str


def _edges_by_field(data_point):
    return {name: edge for name, edge in data_point.get_edges_from_fields()}


def _leftover_names(data_point):
    return {name for name, _ in data_point.get_fields_without_edges()}


def _leftovers(data_point):
    return dict(data_point.get_fields_without_edges())


def test_nested_datapoint_and_list_are_named_after_the_field():
    car = Car(name="Beetle")
    other = Car(name="Golf")
    person = Owner(name="Alice", owns=car, cars=[other])

    by_name = _edges_by_field(person)
    leftovers = _leftover_names(person)

    assert by_name["owns"].source is person
    assert by_name["owns"].target is car
    assert by_name["owns"].relationship_type == "owns"
    assert by_name["cars"].source is person
    assert by_name["cars"].target is other
    assert by_name["cars"].relationship_type == "cars"
    assert "owns" not in leftovers
    assert "cars" not in leftovers
    assert leftovers >= {"name"}


def test_tuple_edge_weight_and_purchased_name():
    car = Car(name="Beetle")
    person = Owner(name="Alice", purchased=(Edge(weight=0.8), car))

    edge = _edges_by_field(person)["purchased"]
    assert edge.relationship_type == "purchased"
    assert edge.weight == 0.8
    assert "purchased" not in _leftover_names(person)

    person = Owner(name="Alice", purchased=(Edge(relationship_type="bought"), car))
    edge = _edges_by_field(person)["purchased"]
    assert edge.relationship_type == "bought"


def test_local_edge_fills_source_from_owner():
    car = Car(name="Beetle")
    person = Owner(name="Alice", extra=Edge(target=car, weight=0.8))

    edge = _edges_by_field(person)["extra"]
    assert edge.source is person
    assert edge.target is car
    assert edge.relationship_type == "extra"
    assert edge.weight == 0.8
    assert "extra" not in _leftover_names(person)


def test_explicit_edge_keeps_foreign_endpoints():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    graph = SocialGraph(friends_with=[Edge(source=alice, target=bob)])

    edge = _edges_by_field(graph)["friends_with"]
    assert edge.source is alice
    assert edge.target is bob
    assert graph not in (edge.source, edge.target)
    assert "friends_with" not in _leftover_names(graph)


def test_edge_without_target_stays_a_leftover_field():
    person = Owner(name="Alice", extra=Edge(weight=0.8))
    assert person.get_edges_from_fields() == []
    assert _leftovers(person)["extra"].weight == 0.8


def test_empty_tuple_targets_expand_to_nothing():
    person = Owner(name="Alice", empty_tuple=(Edge(weight=0.8), []))
    assert person.get_edges_from_fields() == []
    assert "empty_tuple" in _leftover_names(person)


def test_normalize_keeps_mentors_on_verbatim():
    car = Car(name="Beetle")
    person = Owner(name="Alice", mentors=(Edge(relationship_type="Mentors On"), car))
    edge = _edges_by_field(person)["mentors"]
    assert edge.relationship_type == "Mentors On"


def test_belongs_to_set_expands_to_edges_and_is_not_a_leftover():
    node_set = NodeSet(name="team")
    person = Owner(name="Alice", belongs_to_set=[node_set])
    assert "belongs_to_set" not in _leftover_names(person)
    assert _edges_by_field(person)["belongs_to_set"].target is node_set


def test_metadata_is_on_neither_list():
    person = Owner(name="Alice")
    assert "metadata" not in _leftover_names(person)
    assert all(name != "metadata" for name, _ in person.get_edges_from_fields())


def test_transparent_target_stays_raw():
    alice = Person(name="Alice")
    group = Group(members=[alice])
    holder = Owner(name="Dept", groups=[group])
    edge = _edges_by_field(holder)["groups"]
    assert edge.target is group


def test_model_with_no_edges():
    plain = Plain(name="x")
    assert plain.get_edges_from_fields() == []
    assert "name" in _leftover_names(plain)


def test_tuple_target_argument_wins():
    car = Car(name="Beetle")
    other = Car(name="Golf")
    person = Owner(name="Alice", mixed=(Edge(target=car), other))
    edge = _edges_by_field(person)["mixed"]
    assert edge.target is other


def test_datapoint_imports_in_a_fresh_interpreter():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cognee.infrastructure.engine.models.DataPoint import DataPoint",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
