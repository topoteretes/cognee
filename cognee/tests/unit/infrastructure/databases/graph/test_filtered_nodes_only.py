"""Filtered node reads must not issue the discarded edge query."""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name,class_name,row",
    [
        (
            "ladybug",
            "LadybugAdapter",
            ("conflict", {"type": "FactConflict", "properties": '{"selected_fact": "Accepted"}'}),
        ),
        (
            "neo4j_driver",
            "Neo4jAdapter",
            {"id": "conflict", "properties": {"type": "FactConflict", "selected_fact": "Accepted"}},
        ),
        (
            "neptune_driver",
            "NeptuneGraphDB",
            {"id": "conflict", "properties": {"type": "FactConflict", "selected_fact": "Accepted"}},
        ),
    ],
)
async def test_cypher_adapters_skip_edge_queries(module_name, class_name, row):
    if module_name == "neo4j_driver":
        pytest.importorskip("neo4j")
    elif module_name == "neptune_driver":
        pytest.importorskip("botocore")
    module = importlib.import_module(f"cognee.infrastructure.databases.graph.{module_name}.adapter")
    adapter = object.__new__(getattr(module, class_name))
    adapter.query = AsyncMock(return_value=[row])
    nodes, edges = await adapter.get_filtered_graph_data(
        [{"type": ["FactConflict"]}], include_edges=False
    )
    assert nodes[0][1]["selected_fact"] == "Accepted"
    assert edges == []
    assert adapter.query.await_count == 1
    assert "-[" not in adapter.query.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name,class_name", [("postgres_demo", "PostgresDemoAdapter"), ("turso", "TursoAdapter")]
)
@pytest.mark.parametrize("filters", [[], [{"type": ["FactConflict"]}]])
async def test_sql_adapters_skip_edge_queries(module_name, class_name, filters):
    module = importlib.import_module(f"cognee.infrastructure.databases.graph.{module_name}.adapter")
    adapter = object.__new__(getattr(module, class_name))
    result = Mock()
    result.mappings.return_value.all.return_value = [
        {
            "id": "conflict",
            "name": "",
            "type": "FactConflict",
            "properties": {"selected_fact": "Accepted"},
        }
    ]
    result.fetchall.return_value = [("conflict", "", "FactConflict", {"selected_fact": "Accepted"})]
    session = SimpleNamespace(execute=AsyncMock(return_value=result))

    @asynccontextmanager
    async def session_scope():
        yield session

    if module_name == "turso":
        adapter._session = session_scope
    else:
        adapter.sessionmaker = session_scope
    nodes, edges = await adapter.get_filtered_graph_data(filters, include_edges=False)
    assert nodes[0][1]["selected_fact"] == "Accepted"
    assert edges == []
    assert session.execute.await_count == 1
    assert "graph_edge" not in str(session.execute.await_args.args[0])
