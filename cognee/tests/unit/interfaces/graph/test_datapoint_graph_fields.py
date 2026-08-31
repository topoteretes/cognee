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


def _declared_by_name(declared):
    return {name: edge for name, edge in declared}


def test_nested_datapoint_and_list_are_named_after_the_field():
    car = Car(name="Beetle")
    other = Car(name="Golf")
    person = Owner(name="Alice", owns=car, cars=[other])

    properties, excluded, declared = person.graph_fields()
    by_name = _declared_by_name(declared)

    assert by_name["owns"].source is person
    assert by_name["owns"].target is car
    assert by_name["owns"].relationship_type == "owns"
    assert by_name["cars"].source is person
    assert by_name["cars"].target is other
    assert by_name["cars"].relationship_type == "cars"
    assert "owns" in excluded
    assert "cars" in excluded
    assert "owns" not in properties
    assert "cars" not in properties


def test_tuple_edge_weight_and_purchased_name():
    car = Car(name="Beetle")
    person = Owner(name="Alice", purchased=(Edge(weight=0.8), car))

    _, excluded, declared = person.graph_fields()
    edge = _declared_by_name(declared)["purchased"]
    assert edge.relationship_type == "purchased"
    assert edge.weight == 0.8
    assert "purchased" in excluded

    person = Owner(name="Alice", purchased=(Edge(relationship_type="bought"), car))
    edge = _declared_by_name(person.graph_fields()[2])["purchased"]
    assert edge.relationship_type == "bought"


def test_local_edge_fills_source_from_owner():
    car = Car(name="Beetle")
    person = Owner(name="Alice", extra=Edge(target=car, weight=0.8))

    _, excluded, declared = person.graph_fields()
    edge = _declared_by_name(declared)["extra"]
    assert edge.source is person
    assert edge.target is car
    assert edge.relationship_type == "extra"
    assert edge.weight == 0.8
    assert "extra" in excluded


def test_explicit_edge_keeps_foreign_endpoints():
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    graph = SocialGraph(friends_with=[Edge(source=alice, target=bob)])

    _, excluded, declared = graph.graph_fields()
    edge = _declared_by_name(declared)["friends_with"]
    assert edge.source is alice
    assert edge.target is bob
    assert graph not in (edge.source, edge.target)
    assert "friends_with" in excluded


def test_edge_without_target_stays_a_property():
    person = Owner(name="Alice", extra=Edge(weight=0.8))
    properties, excluded, declared = person.graph_fields()
    assert declared == []
    assert "extra" not in excluded
    assert properties["extra"].weight == 0.8


def test_empty_tuple_targets_declare_nothing():
    person = Owner(name="Alice", empty_tuple=(Edge(weight=0.8), []))
    properties, excluded, declared = person.graph_fields()
    assert declared == []
    assert "empty_tuple" not in excluded
    assert "empty_tuple" in properties


def test_normalize_keeps_mentors_on_verbatim():
    car = Car(name="Beetle")
    person = Owner(name="Alice", mentors=(Edge(relationship_type="Mentors On"), car))
    edge = _declared_by_name(person.graph_fields()[2])["mentors"]
    assert edge.relationship_type == "Mentors On"


def test_belongs_to_set_is_stored_and_declared():
    node_set = NodeSet(name="team")
    person = Owner(name="Alice", belongs_to_set=[node_set])
    properties, excluded, declared = person.graph_fields()
    assert "belongs_to_set" not in excluded
    assert properties["belongs_to_set"] == ["team"]
    assert _declared_by_name(declared)["belongs_to_set"].target is node_set


def test_metadata_is_in_none_of_the_three():
    person = Owner(name="Alice")
    properties, excluded, declared = person.graph_fields()
    assert "metadata" not in properties
    assert "metadata" not in excluded
    assert all(name != "metadata" for name, _ in declared)


def test_transparent_target_stays_raw():
    alice = Person(name="Alice")
    group = Group(members=[alice])
    holder = Owner(name="Dept", groups=[group])
    edge = _declared_by_name(holder.graph_fields()[2])["groups"]
    assert edge.target is group


def test_model_with_no_relationships():
    plain = Plain(name="x")
    _properties, excluded, declared = plain.graph_fields()
    assert excluded == set()
    assert declared == []


def test_tuple_target_argument_wins():
    car = Car(name="Beetle")
    other = Car(name="Golf")
    person = Owner(name="Alice", mixed=(Edge(target=car), other))
    edge = _declared_by_name(person.graph_fields()[2])["mixed"]
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
