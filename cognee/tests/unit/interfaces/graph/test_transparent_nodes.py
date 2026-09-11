"""Transparency: wherever a transparent node appears, it is replaced by its children.

Edges are compared as sets: the walk emits them in field-declaration order, which is
not part of the contract.
"""

import logging
from typing import Any, List, Optional

import pytest

from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.engine.models import NodeSet
from cognee.modules.graph.utils import get_graph_from_model
from cognee.modules.graph.utils.unwrap_transparent_nodes import _WARNED_DROPPED_FIELDS

TRANSPARENT = {"index_fields": [], "transparent": True}


class Activity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Company(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    likes: Optional[List[Activity]] = None
    works_for: Optional[Company] = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Directory(DataPoint):
    people: List[Person]
    companies: List[Company]
    metadata: dict = {"index_fields": []}


class TransparentDirectory(DataPoint):
    people: List[Person]
    companies: List[Company]
    metadata: dict = TRANSPARENT


class MemberGroup(DataPoint):
    members: List[Person]
    metadata: dict = TRANSPARENT


class NestedDirectory(DataPoint):
    groups: List[MemberGroup]
    companies: List[Company]
    metadata: dict = TRANSPARENT


class Department(DataPoint):
    name: str
    groups: List[Any]
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class NamedGroup(DataPoint):
    """A transparent wrapper that wrongly carries real data in ``name``."""

    name: str
    members: List[Person]
    metadata: dict = {"index_fields": ["name"], "transparent": True}


class EmptyGroup(DataPoint):
    members: List[Person] = []
    metadata: dict = TRANSPARENT


class HolderOfEmpty(DataPoint):
    name: str
    groups: List[EmptyGroup]
    metadata: dict = {"index_fields": ["name"]}


class DiamondHolder(DataPoint):
    name: str
    left: List[Any]
    right: List[Any]
    metadata: dict = {"index_fields": ["name"]}


class OptionalGroup(DataPoint):
    """Relationship-only wrapper whose optional fields are legitimately empty."""

    members: List[Person] = []
    lead: Optional[Person] = None
    metadata: dict = TRANSPARENT


class Chunk(DataPoint):
    text: str
    contains: Any = None
    metadata: dict = {"index_fields": ["text"]}


def _people():
    acme = Company(name="Acme")
    return (
        Person(name="Alice", likes=[Activity(name="Biking")], works_for=acme),
        Person(name="Bob", likes=[Activity(name="Basketball")], works_for=acme),
        acme,
    )


def _edge_set(edges):
    return {(str(source), str(target), name) for source, target, name, _ in edges}


def _ids(nodes):
    return {str(node.id) for node in nodes}


@pytest.mark.asyncio
async def test_plain_wrapper_is_stored_as_a_node():
    """Case 1 (A1): a wrapper with no flag keeps today's graph exactly."""
    alice, bob, acme = _people()
    directory = Directory(people=[alice, bob], companies=[acme])

    nodes, edges = await get_graph_from_model(directory)

    assert str(directory.id) in _ids(nodes)
    assert _edge_set(edges) == {
        (str(directory.id), str(alice.id), "people"),
        (str(directory.id), str(bob.id), "people"),
        (str(directory.id), str(acme.id), "companies"),
        (str(alice.id), str(alice.likes[0].id), "likes"),
        (str(alice.id), str(acme.id), "works_for"),
        (str(bob.id), str(bob.likes[0].id), "likes"),
        (str(bob.id), str(acme.id), "works_for"),
    }


@pytest.mark.asyncio
async def test_transparent_root_stores_children_as_top_level():
    """Case 2 (A2): the wrapper is absent from nodes and from both edge endpoints."""
    alice, bob, acme = _people()
    directory = TransparentDirectory(people=[alice, bob], companies=[acme])

    nodes, edges = await get_graph_from_model(directory)

    wrapper_id = str(directory.id)
    assert wrapper_id not in _ids(nodes)
    assert _ids(nodes) == {
        str(alice.id),
        str(bob.id),
        str(acme.id),
        str(alice.likes[0].id),
        str(bob.likes[0].id),
    }
    assert _edge_set(edges) == {
        (str(alice.id), str(alice.likes[0].id), "likes"),
        (str(alice.id), str(acme.id), "works_for"),
        (str(bob.id), str(bob.likes[0].id), "likes"),
        (str(bob.id), str(acme.id), "works_for"),
    }
    assert all(wrapper_id not in (str(source), str(target)) for source, target, _, _ in edges)


@pytest.mark.asyncio
async def test_nested_transparent_wrappers_resolve_recursively():
    """Case 3 (A3): identical to A2 after applying the rule twice."""
    alice, bob, acme = _people()
    flat = TransparentDirectory(people=[alice, bob], companies=[acme])
    nested = NestedDirectory(groups=[MemberGroup(members=[alice, bob])], companies=[acme])

    flat_nodes, flat_edges = await get_graph_from_model(flat)
    nested_nodes, nested_edges = await get_graph_from_model(nested)

    assert _ids(nested_nodes) == _ids(flat_nodes)
    assert _edge_set(nested_edges) == _edge_set(flat_edges)


@pytest.mark.asyncio
async def test_mid_graph_wrapper_keeps_parent_field_name():
    """Case 4 (A4): the ``groups`` edge lands on each child."""
    alice, bob, acme = _people()
    department = Department(name="Engineering", groups=[MemberGroup(members=[alice, bob])])

    nodes, edges = await get_graph_from_model(department)

    assert str(department.id) in _ids(nodes)
    assert (str(department.id), str(alice.id), "groups") in _edge_set(edges)
    assert (str(department.id), str(bob.id), "groups") in _edge_set(edges)
    assert not any(str(node.id) == str(department.groups[0].id) for node in nodes)


@pytest.mark.asyncio
async def test_edge_metadata_is_applied_to_each_child():
    """Case 13: an ``(Edge, wrapper)`` tuple re-points with its metadata intact."""
    alice, bob, _ = _people()
    edge = Edge(relationship_type="staffed_by", weights={"strength": 0.5})
    department = Department(name="Engineering", groups=[(edge, MemberGroup(members=[alice, bob]))])

    _, edges = await get_graph_from_model(department)

    repointed = [
        properties
        for _, target, _, properties in edges
        if str(target) in (str(alice.id), str(bob.id))
    ]
    assert len(repointed) == 2
    for properties in repointed:
        assert properties["relationship_name"] == "staffed_by"
        assert properties["weight_strength"] == 0.5


@pytest.mark.asyncio
async def test_wrapper_carrying_scalar_data_warns_once_and_drops_it(caplog):
    """Case 5 (A5): the value is dropped, no node is minted, one warning is logged."""
    _WARNED_DROPPED_FIELDS.clear()
    alice, bob, _ = _people()
    department = Department(
        name="Engineering", groups=[NamedGroup(name="Core team", members=[alice, bob])]
    )

    with caplog.at_level(logging.WARNING):
        nodes, edges = await get_graph_from_model(department)

    assert str(department.groups[0].id) not in _ids(nodes)
    assert not any("Core team" in str(getattr(node, "name", "")) for node in nodes)
    assert (str(department.id), str(alice.id), "groups") in _edge_set(edges)

    warnings = [record for record in caplog.records if "marked transparent" in record.getMessage()]
    assert len(warnings) == 1
    assert "'name'" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_dropped_field_warning_fires_at_most_once(caplog):
    _WARNED_DROPPED_FIELDS.clear()
    alice, bob, _ = _people()

    with caplog.at_level(logging.WARNING):
        for label in ("Core team", "Platform team"):
            await get_graph_from_model(
                Department(name=label, groups=[NamedGroup(name=label, members=[alice, bob])])
            )

    warnings = [record for record in caplog.records if "marked transparent" in record.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_relationship_only_wrapper_never_warns(caplog):
    """An optional relationship left ``None`` and an empty list lose nothing."""
    _WARNED_DROPPED_FIELDS.clear()
    alice, _, _ = _people()

    with caplog.at_level(logging.WARNING):
        await get_graph_from_model(
            Department(name="Engineering", groups=[OptionalGroup(members=[alice], lead=None)])
        )
        await get_graph_from_model(
            Department(name="Support", groups=[OptionalGroup(members=[], lead=None)])
        )

    assert not [record for record in caplog.records if "marked transparent" in record.getMessage()]


@pytest.mark.asyncio
async def test_empty_wrapper_as_root_returns_nothing():
    """Case 6."""
    assert await get_graph_from_model(EmptyGroup(members=[])) == ([], [])


@pytest.mark.asyncio
async def test_field_holding_only_empty_wrappers_is_not_a_property():
    """Case 7: no edge is minted and the field never becomes a scalar property."""
    holder = HolderOfEmpty(name="Holder", groups=[EmptyGroup(members=[])])

    nodes, edges = await get_graph_from_model(holder)

    assert _ids(nodes) == {str(holder.id)}
    assert edges == []
    assert not hasattr(nodes[0], "groups")


@pytest.mark.asyncio
async def test_transparent_via_constructor_kwarg_is_honoured():
    """Case 8: the MetaData TypedDict must declare ``transparent`` for this to survive."""
    alice, bob, acme = _people()
    directory = Directory(
        people=[alice, bob],
        companies=[acme],
        metadata={"index_fields": [], "transparent": True},
    )

    assert directory.metadata.get("transparent") is True

    nodes, _ = await get_graph_from_model(directory)

    assert str(directory.id) not in _ids(nodes)


@pytest.mark.asyncio
async def test_children_across_several_fields_appear_exactly_once():
    """Case 9."""
    alice, bob, acme = _people()
    directory = TransparentDirectory(people=[alice, bob, alice], companies=[acme])

    nodes, _ = await get_graph_from_model(directory)

    ids = [str(node.id) for node in nodes]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_diamond_wrapper_reached_from_two_fields_keeps_its_children():
    """Case 10: regression guard for a shared mutable ``_active`` set."""
    alice, bob, _ = _people()
    group = MemberGroup(members=[alice, bob])
    holder = DiamondHolder(name="Holder", left=[group], right=[group])

    _, edges = await get_graph_from_model(holder)

    assert (str(holder.id), str(alice.id), "left") in _edge_set(edges)
    assert (str(holder.id), str(alice.id), "right") in _edge_set(edges)
    assert (str(holder.id), str(bob.id), "right") in _edge_set(edges)


@pytest.mark.asyncio
async def test_distinct_instances_sharing_a_node_id_both_resolve():
    """Case 11: identity_fields collision must not read as a cycle."""

    class IdentityGroup(DataPoint):
        name: str
        members: List[Person]
        metadata: dict = {
            "index_fields": ["name"],
            "identity_fields": ["name"],
            "transparent": True,
        }

    alice, bob, _ = _people()
    first = IdentityGroup(name="Team", members=[alice])
    second = IdentityGroup(name="Team", members=[bob])
    assert first.id == second.id

    holder = DiamondHolder(name="Holder", left=[first], right=[second])

    _, edges = await get_graph_from_model(holder)

    assert (str(holder.id), str(alice.id), "left") in _edge_set(edges)
    assert (str(holder.id), str(bob.id), "right") in _edge_set(edges)


@pytest.mark.asyncio
async def test_shared_child_of_two_parents_is_stored_once():
    """Case 12."""
    alice, bob, acme = _people()
    directory = TransparentDirectory(people=[alice, bob], companies=[acme])

    nodes, edges = await get_graph_from_model(directory)

    assert [str(node.id) for node in nodes].count(str(acme.id)) == 1
    assert (str(alice.id), str(acme.id), "works_for") in _edge_set(edges)
    assert (str(bob.id), str(acme.id), "works_for") in _edge_set(edges)


@pytest.mark.asyncio
async def test_belongs_to_set_is_not_a_wrapper_child():
    """Case 14: a wrapper's NodeSets must not be inherited by its parent."""
    alice, _, _ = _people()
    node_set = NodeSet(name="team")
    group = MemberGroup(members=[alice])
    group.belongs_to_set = [node_set]
    department = Department(name="Engineering", groups=[group])

    nodes, edges = await get_graph_from_model(department)

    assert str(node_set.id) not in _ids(nodes)
    assert all(str(target) != str(node_set.id) for _, target, _, _ in edges)


@pytest.mark.asyncio
async def test_transparent_only_cycle_terminates():
    """Case 15."""

    class CyclicGroup(DataPoint):
        peers: List[Any] = []
        metadata: dict = TRANSPARENT

    first = CyclicGroup()
    second = CyclicGroup(peers=[first])
    first.peers = [second]

    assert await get_graph_from_model(first) == ([], [])


@pytest.mark.asyncio
async def test_mixed_cycle_terminates_with_a_self_edge():
    """Case 16: transparent -> plain -> transparent collapses onto the plain node."""

    class Mixed(DataPoint):
        name: str
        groups: List[Any] = []
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class Wrapper(DataPoint):
        members: List[Any] = []
        metadata: dict = TRANSPARENT

    person = Mixed(name="Alice")
    wrapper = Wrapper(members=[person])
    person.groups = [wrapper]

    nodes, edges = await get_graph_from_model(wrapper)

    assert _ids(nodes) == {str(person.id)}
    assert _edge_set(edges) == {(str(person.id), str(person.id), "groups")}


@pytest.mark.asyncio
async def test_chunk_contains_a_transparent_wrapper():
    """Case 18 (B3): the cognify chunk boundary, assigned after construction."""
    alice, bob, acme = _people()
    chunk = Chunk(text="Alice works for Acme. Bob works for Acme.")
    chunk.contains = TransparentDirectory(people=[alice, bob], companies=[acme])
    wrapper_id = str(chunk.contains.id)

    nodes, edges = await get_graph_from_model(chunk)

    assert wrapper_id not in _ids(nodes)
    contains = {str(target) for source, target, name, _ in edges if name == "contains"}
    assert contains == {str(alice.id), str(bob.id), str(acme.id)}
    assert all(str(source) == str(chunk.id) for source, _, name, _ in edges if name == "contains")


def test_transparent_survives_every_metadata_form():
    """All three rows of the plan's metadata table keep the flag."""
    alice, bob, acme = _people()

    class Flagged(DataPoint):
        people: List[Person]
        metadata: dict = {"index_fields": [], "transparent": True}

    assert Flagged(people=[alice]).metadata.get("transparent") is True

    from_kwarg = Directory(
        people=[alice, bob],
        companies=[acme],
        metadata={"index_fields": [], "transparent": True},
    )
    assert from_kwarg.metadata.get("transparent") is True

    assigned = Directory(people=[alice], companies=[acme])
    assigned.metadata = {"index_fields": [], "transparent": True}
    assert assigned.metadata.get("transparent") is True

    revalidated = Directory.model_validate(from_kwarg.model_dump())
    assert revalidated.metadata.get("transparent") is True


class CyclicPerson(DataPoint):
    name: str
    works_for: Any = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class CyclicCompany(DataPoint):
    name: str
    employees: List[Any] = []
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


def _parity_fixtures():
    """One root per storage shape the collector has to agree with."""
    plain_a, plain_b, plain_acme = _people()
    transparent_a, transparent_b, transparent_acme = _people()
    nested_a, nested_b, nested_acme = _people()
    node_set_a, _, _ = _people()

    with_node_set = Department(name="Engineering", groups=[MemberGroup(members=[node_set_a])])
    with_node_set.belongs_to_set = [NodeSet(name="team")]

    person = CyclicPerson(name="Cyclic Alice")
    company = CyclicCompany(name="Cyclic Acme", employees=[person])
    person.works_for = company

    return [
        Directory(people=[plain_a, plain_b], companies=[plain_acme]),
        TransparentDirectory(people=[transparent_a, transparent_b], companies=[transparent_acme]),
        NestedDirectory(
            groups=[MemberGroup(members=[nested_a, nested_b])], companies=[nested_acme]
        ),
        with_node_set,
        company,
    ]


@pytest.mark.parametrize("root", _parity_fixtures())
@pytest.mark.asyncio
async def test_collector_returns_the_originals_that_storage_writes(root):
    """Same node set as the walk, but the original objects rather than the copies."""
    from cognee.modules.graph.utils import collect_stored_data_points

    stored_nodes, _ = await get_graph_from_model(root)
    collected = await collect_stored_data_points(root)

    assert {str(node.id) for node in collected} == _ids(stored_nodes)
    # The walk's own output is unusable for linking: copy_model strips the
    # relationship fields, so edges out of those nodes would never be minted.
    assert all(isinstance(node, DataPoint) for node in collected)
    assert all(type(node) is not type(copy) for node, copy in zip(collected, stored_nodes))
