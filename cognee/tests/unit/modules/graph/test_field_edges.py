import asyncio
import subprocess
import sys

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import NodeSet
from cognee.modules.graph.utils.field_edges import (
    get_edges_from_fields,
    get_fields_without_edges,
)
from cognee.modules.storage.utils import copy_model


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
    return {name: edge for name, edge in get_edges_from_fields(data_point)}


def _leftover_names(data_point):
    return {name for name, _ in get_fields_without_edges(data_point)}


def _leftovers(data_point):
    return dict(get_fields_without_edges(data_point))


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
    assert get_edges_from_fields(person) == []
    assert _leftovers(person)["extra"].weight == 0.8


def test_empty_tuple_targets_expand_to_nothing():
    person = Owner(name="Alice", empty_tuple=(Edge(weight=0.8), []))
    assert get_edges_from_fields(person) == []
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
    assert all(name != "metadata" for name, _ in get_edges_from_fields(person))


def test_transparent_target_stays_raw():
    alice = Person(name="Alice")
    group = Group(members=[alice])
    holder = Owner(name="Dept", groups=[group])
    edge = _edges_by_field(holder)["groups"]
    assert edge.target is group


def test_model_with_no_edges():
    plain = Plain(name="x")
    assert get_edges_from_fields(plain) == []
    assert "name" in _leftover_names(plain)


def test_tuple_target_argument_wins():
    car = Car(name="Beetle")
    other = Car(name="Golf")
    person = Owner(name="Alice", mixed=(Edge(target=car), other))
    edge = _edges_by_field(person)["mixed"]
    assert edge.target is other


def test_field_edges_read_a_plain_copy_of_a_datapoint():
    """``copy_model`` mints plain BaseModel subclasses, and those get walked too.

    A chunk rebuilt from an export is one. It is not a DataPoint and may be missing
    fields the class declares, but it still holds real DataPoint children whose edges
    have to be emitted — classification cannot require the owner to be a DataPoint.
    """
    SimpleOwner = copy_model(Owner, exclude_fields=["cars"])
    car = Car(name="Beetle")
    owner = SimpleOwner(name="Alice", owns=car)

    assert not isinstance(owner, DataPoint)
    assert not hasattr(owner, "cars")

    field_edges = get_edges_from_fields(owner)

    assert [name for name, _ in field_edges] == ["owns"]
    assert field_edges[0][1].source is owner
    assert field_edges[0][1].target is car
    assert "name" in {name for name, _ in get_fields_without_edges(owner)}
    assert "owns" not in {name for name, _ in get_fields_without_edges(owner)}


def test_every_walk_entry_point_accepts_a_plain_copy():
    """The three callers each reached for a DataPoint method, so each could break alone.

    Chunk ownership covered only ``get_graph_from_model``; the crash reached
    ``collect_stored_data_points`` and ``unwrap_transparent`` as well.
    """
    from cognee.modules.graph.utils.get_graph_from_model import (
        collect_stored_data_points,
        get_graph_from_model,
    )
    from cognee.modules.graph.utils.unwrap_transparent_nodes import unwrap_transparent

    SimpleOwner = copy_model(Owner, exclude_fields=["cars"])
    car = Car(name="Beetle")
    owner = SimpleOwner(name="Alice", owns=car)

    _nodes, edges = asyncio.run(get_graph_from_model(owner))
    assert [relationship for _s, _t, relationship, _p in edges] == ["owns"]

    stored = asyncio.run(collect_stored_data_points(owner))
    assert owner in stored
    assert car in stored

    assert unwrap_transparent(owner) == [owner]


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


def test_mixed_tuple_targets_only_emit_datapoint_edges():
    """A loosely-typed field can hold a mixed list; non-DataPoint items are skipped."""
    car = Car(name="Beetle")
    person = Owner(name="Alice", purchased=(Edge(weight=0.8), [car, "junk"]))

    edges = [edge for name, edge in get_edges_from_fields(person) if name == "purchased"]

    assert [edge.target for edge in edges] == [car]


def test_targetless_edge_carrying_a_datapoint_is_dropped_from_node_properties(caplog):
    """A property cannot hold a serialized node; storing it would lose the node inside."""
    import logging

    from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model

    car = Car(name="Beetle")
    person = Owner(name="Alice", extra=Edge(source=car, weight=0.8))

    with caplog.at_level(logging.WARNING):
        nodes, _edges = asyncio.run(get_graph_from_model(person))

    assert nodes[0].extra is None
    assert any("cannot be stored as a node property" in r.getMessage() for r in caplog.records)
