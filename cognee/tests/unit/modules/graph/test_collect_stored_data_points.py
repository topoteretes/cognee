import importlib

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import collect_stored_data_points

walk_module = importlib.import_module("cognee.modules.graph.utils.get_graph_from_model")


class Employee(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Company(DataPoint):
    name: str
    employees: list[Employee] = []
    metadata: dict = {"index_fields": ["name"]}


class Chunk(DataPoint):
    text: str
    company: Company
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
async def test_roots_that_share_a_node_walk_it_once(monkeypatch):
    """A batch's chunks all link to the company a document mentions, and walking it again
    for each chunk made the work grow with the square of the chunk count."""
    company = Company(name="acme", employees=[Employee(name=f"e{i}") for i in range(20)])
    chunks = [Chunk(text=f"chunk {i}", company=company) for i in range(40)]
    serialized = []
    original = walk_module._graph_node_from
    monkeypatch.setattr(
        walk_module,
        "_graph_node_from",
        lambda data_point, *args: serialized.append(data_point) or original(data_point, *args),
    )

    stored = await collect_stored_data_points(*chunks)

    assert len(stored) == 40 + 1 + 20
    assert len(serialized) == len(stored)
    assert sum(1 for data_point in stored if data_point is company) == 1
