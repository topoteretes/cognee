"""Custom graph_model nodes carry their chunks' node sets through to a real Ladybug graph.

Runs the extraction integration step and the storage walk for real, writes the result
into an embedded Ladybug database, and reads it back — no LLM and no embeddings. Two
"sources" with different node sets extract the same identity node, the way separate
add + cognify runs into one dataset do (SDK-801).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.utils import ensure_default_edge_properties, get_graph_from_model
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LADYBUG, reason="ladybug not installed"),
]


class Team(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    member_of: Team | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Company(DataPoint):
    people: list[Person]
    metadata: dict = {"index_fields": [], "transparent": True}


class _Chunk(DataPoint):
    """A DocumentChunk-shaped node, so the real walk mints the contains edges."""

    text: str
    contains: Any = None
    metadata: dict = {"index_fields": ["text"]}


def _resolver():
    resolver = MagicMock()
    resolver.get_subgraph.return_value = ([], [], None)
    return resolver


async def _cognify_into(graph, node_set: str, extraction: Company) -> None:
    """Integrate one extraction under ``node_set`` and store it, as one cognify run would."""
    chunk = _Chunk(
        text=node_set, belongs_to_set=[NodeSet(id=NodeSet.id_for(node_set), name=node_set)]
    )
    await integrate_chunk_graphs([chunk], [extraction], Company, _resolver())
    nodes, edges = await get_graph_from_model(chunk)
    await graph.add_nodes(nodes)
    await graph.add_edges(ensure_default_edge_properties(edges, nodes=nodes))


def _hr_extraction() -> Company:
    return Company(people=[Person(name="Dana Kim", member_of=Team(name="Search"))])


def _tickets_extraction() -> Company:
    return Company(people=[Person(name="Dana Kim"), Person(name="Omar Haddad")])


async def test_a_shared_custom_node_keeps_every_node_set(tmp_path):
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        await _cognify_into(graph, "hr_database", _hr_extraction())
        await _cognify_into(graph, "support_tickets", _tickets_extraction())

        dana = await graph.get_node(str(Person.id_for("Dana Kim")))
        omar = await graph.get_node(str(Person.id_for("Omar Haddad")))
        team = await graph.get_node(str(Team.id_for("Search")))

        assert sorted(dana["belongs_to_set"]) == ["hr_database", "support_tickets"]
        assert omar["belongs_to_set"] == ["support_tickets"]
        assert team["belongs_to_set"] == ["hr_database"]
    finally:
        await graph.close()


async def test_node_set_filter_reaches_custom_nodes_and_their_edges(tmp_path):
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        await _cognify_into(graph, "hr_database", _hr_extraction())
        await _cognify_into(graph, "support_tickets", _tickets_extraction())

        nodes, edges = await graph.get_nodeset_subgraph(NodeSet, ["hr_database"])
        names = {properties.get("name") for _, properties in nodes}
        relationships = {name for _, _, name, _ in edges}

        assert {"Dana Kim", "Search"} <= names
        assert "Omar Haddad" not in names
        assert "member_of" in relationships
    finally:
        await graph.close()


async def test_removing_one_node_set_keeps_the_other_on_a_shared_node(tmp_path):
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        await _cognify_into(graph, "hr_database", _hr_extraction())
        await _cognify_into(graph, "support_tickets", _tickets_extraction())

        await graph.remove_belongs_to_set_tags(["hr_database"])

        dana = await graph.get_node(str(Person.id_for("Dana Kim")))
        assert dana is not None
        assert dana["belongs_to_set"] == ["support_tickets"]
    finally:
        await graph.close()
