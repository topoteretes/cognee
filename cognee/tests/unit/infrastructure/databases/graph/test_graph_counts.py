"""Graph-summary counts must not pay for full graph metrics.

``GET /datasets/graph-summary`` and brains-summary read only ``num_nodes`` and
``num_edges``. They used to get them from ``get_graph_metrics``, which on every
backend also computes connected components (a GDS projection on Neo4j, 1-3 hop
path expansion on Ladybug, a recursive CTE on Turso). ``get_graph_counts``
returns the two counts alone; in-tree adapters implement it with count queries.

The interface default derives the counts from ``get_graph_metrics`` so a
community adapter that predates the method keeps working, which is why it has a
test of its own.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


class _MetricsOnlyAdapter:
    """An adapter with no native get_graph_counts: exercises the inherited default.

    Not a GraphDBInterface subclass, for the same reason as in
    test_top_degree_node_ids.py: the default is invoked unbound against this
    object, which runs exactly the code a real adapter would inherit.
    """

    def __init__(self, metrics):
        self.get_graph_metrics = AsyncMock(return_value=metrics)

    async def get_graph_counts(self) -> tuple[int, int]:
        return await GraphDBInterface.get_graph_counts(self)


@pytest.mark.asyncio
async def test_default_reads_the_counts_from_get_graph_metrics():
    adapter = _MetricsOnlyAdapter({"num_nodes": 5, "num_edges": 8, "num_connected_components": 2})

    assert await adapter.get_graph_counts() == (5, 8)
    adapter.get_graph_metrics.assert_awaited_once_with(include_optional=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("metrics", [{}, None, {"num_nodes": None, "num_edges": None}])
async def test_default_reads_missing_counts_as_zero(metrics):
    """An adapter that omits a key must not put None into an int field."""
    assert await _MetricsOnlyAdapter(metrics).get_graph_counts() == (0, 0)


@pytest.mark.parametrize(
    ("module", "class_name"),
    [
        ("ladybug.adapter", "LadybugAdapter"),
        ("neo4j_driver.adapter", "Neo4jAdapter"),
        ("neptune_driver.adapter", "NeptuneGraphDB"),
        ("postgres_demo.adapter", "PostgresDemoAdapter"),
        ("turso.adapter", "TursoAdapter"),
    ],
)
def test_in_tree_adapters_count_natively(module, class_name):
    """Every in-tree adapter overrides the default, so none falls back to full metrics."""
    # Backends whose driver extra is not installed are skipped, not failed.
    adapter_module = pytest.importorskip(f"cognee.infrastructure.databases.graph.{module}")
    adapter_class = getattr(adapter_module, class_name)
    assert adapter_class.get_graph_counts is not GraphDBInterface.get_graph_counts
