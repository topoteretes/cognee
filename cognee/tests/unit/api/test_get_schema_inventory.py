"""Unit tests for get_schema_inventory over a synthetic knowledge graph."""

from contextlib import asynccontextmanager
from importlib import import_module
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

# Import via importlib: the cognee.api.v1 package rebinds the name "visualize"
# to the visualize_graph function, which shadows attribute-style submodule access.
inventory_module = import_module("cognee.api.v1.visualize.get_schema_inventory")
get_schema_inventory = inventory_module.get_schema_inventory


def _synthetic_graph():
    """Build a (nodes, edges) tuple in the exact verified get_graph_data shape.

    Three Person entities, one Tool, two Brokers. Each instance node carries the
    literal ``type == "Entity"`` and links to its EntityType node via the real
    ``is_a`` edge (Entity -> EntityType). A couple of inter-entity edges exercise
    the relationship distribution.
    """
    nodes = [
        # EntityType nodes (semantic types)
        ("t-person", {"type": "EntityType", "name": "Person"}),
        ("t-tool", {"type": "EntityType", "name": "Tool"}),
        ("t-broker", {"type": "EntityType", "name": "Broker"}),
        # Person entities
        ("e-carlos", {"type": "Entity", "name": "Carlos"}),
        ("e-mika", {"type": "Entity", "name": "Mika"}),
        ("e-sandra", {"type": "Entity", "name": "Sandra"}),
        # Tool entity
        ("e-load-search", {"type": "Entity", "name": "load-search"}),
        # Broker entities
        ("e-echo", {"type": "Entity", "name": "Echo"}),
        ("e-landstar", {"type": "Entity", "name": "Landstar"}),
    ]
    edges = [
        # is_a edges: Entity -> EntityType
        ("e-carlos", "t-person", "is_a", {}),
        ("e-mika", "t-person", "is_a", {}),
        ("e-sandra", "t-person", "is_a", {}),
        ("e-load-search", "t-tool", "is_a", {}),
        ("e-echo", "t-broker", "is_a", {}),
        ("e-landstar", "t-broker", "is_a", {}),
        # Inter-entity relationships
        ("e-carlos", "e-echo", "works_for", {}),
        ("e-mika", "e-landstar", "works_for", {}),
        ("e-carlos", "e-load-search", "uses", {}),
    ]
    return (nodes, edges)


@pytest.fixture
def mock_graph_engine(monkeypatch):
    """Patch get_graph_engine so get_graph_data returns the synthetic graph."""
    engine = AsyncMock()
    engine.get_graph_data = AsyncMock(return_value=_synthetic_graph())
    monkeypatch.setattr(inventory_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


@pytest.mark.asyncio
async def test_counts_are_true_totals(mock_graph_engine):
    inventory = await get_schema_inventory(samples_per_type=2)
    counts = {rec["type"]: rec["count"] for rec in inventory}

    # Counts reflect every instance, not just sampled ones
    assert counts["Person"] == 3
    assert counts["Broker"] == 2
    assert counts["Tool"] == 1

    # The literal "Entity" type is never reported; semantic types are.
    # Internal EntityType taxonomy nodes are resolved away and not reported.
    assert "Entity" not in counts
    assert "EntityType" not in counts


@pytest.mark.asyncio
async def test_samples_capped_and_drawn_from_type(mock_graph_engine):
    inventory = await get_schema_inventory(samples_per_type=2)
    by_type = {rec["type"]: rec for rec in inventory}

    person = by_type["Person"]
    assert person["sample_size"] == min(2, person["count"]) == 2
    assert len(person["samples"]) == 2
    assert set(person["samples"]).issubset({"Carlos", "Mika", "Sandra"})

    tool = by_type["Tool"]
    assert tool["sample_size"] == 1
    assert tool["samples"] == ["load-search"]


@pytest.mark.asyncio
async def test_samples_are_degree_ranked(mock_graph_engine):
    inventory = await get_schema_inventory(samples_per_type=1)
    person = next(rec for rec in inventory if rec["type"] == "Person")

    # Carlos has 3 edges (is_a, works_for, uses); Mika 2; Sandra 1.
    # Highest degree wins the single sample slot.
    assert person["samples"] == ["Carlos"]


@pytest.mark.asyncio
async def test_relationship_distribution(mock_graph_engine):
    inventory = await get_schema_inventory()
    by_type = {rec["type"]: rec for rec in inventory}

    person_rels = by_type["Person"]["relationships"]
    # Person -> Broker via works_for appears twice (Carlos->Echo, Mika->Landstar)
    works_for = next(
        rel for rel in person_rels if rel["relation"] == "works_for" and rel["to_type"] == "Broker"
    )
    assert works_for["count"] == 2

    # Person -> Tool via uses appears once (Carlos->load-search)
    uses = next(rel for rel in person_rels if rel["relation"] == "uses")
    assert uses == {"to_type": "Tool", "relation": "uses", "count": 1}

    # Internal is_a taxonomy edges are resolved away from the public inventory.
    assert all(rel["relation"] != "is_a" for rel in person_rels)


@pytest.mark.asyncio
async def test_sort_by_count_descending(mock_graph_engine):
    inventory = await get_schema_inventory(sort="count")
    counts = [rec["count"] for rec in inventory]
    assert counts == sorted(counts, reverse=True)

    # Person (3) is the top type; type name breaks ties on equal counts.
    ordering = [(rec["count"], rec["type"]) for rec in inventory]
    assert ordering == sorted(ordering, key=lambda pair: (-pair[0], pair[1]))


@pytest.mark.asyncio
async def test_dataset_none_skips_scoping(mock_graph_engine, monkeypatch):
    """The default dataset=None path never enters the scoping context manager."""
    scope = AsyncMock(side_effect=AssertionError("scoping must not run for dataset=None"))
    monkeypatch.setattr(inventory_module, "set_database_global_context_variables", scope)

    inventory = await get_schema_inventory(dataset=None, samples_per_type=2)
    assert len(inventory) > 0


@pytest.mark.asyncio
async def test_dataset_string_does_not_crash(mock_graph_engine, monkeypatch):
    """String dataset names cannot resolve to an owner_id; scoping is skipped safely."""
    # A string name resolves to owner_id=None, which must not reach the context manager.
    scope = AsyncMock(side_effect=AssertionError("scoping must not run for None owner_id"))
    monkeypatch.setattr(inventory_module, "set_database_global_context_variables", scope)

    inventory = await get_schema_inventory(dataset="my-dataset-name", samples_per_type=2)
    assert len(inventory) > 0


@pytest.mark.asyncio
async def test_dataset_uuid_scopes_databases(mock_graph_engine, monkeypatch):
    """A resolvable UUID dataset enters set_database_global_context_variables."""
    dataset_id = uuid4()
    owner_id = uuid4()

    monkeypatch.setattr(
        inventory_module,
        "_resolve_dataset_owner",
        AsyncMock(return_value=owner_id),
    )

    # Record the scoping call and provide a working async context manager.
    calls = []

    @asynccontextmanager
    async def fake_scope(dataset, user_id):
        calls.append((dataset, user_id))
        yield

    monkeypatch.setattr(inventory_module, "set_database_global_context_variables", fake_scope)

    inventory = await get_schema_inventory(dataset=dataset_id, samples_per_type=2)

    assert calls == [(dataset_id, owner_id)]
    assert len(inventory) > 0
