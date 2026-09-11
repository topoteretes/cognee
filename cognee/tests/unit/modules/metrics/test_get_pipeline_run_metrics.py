"""get_pipeline_run_metrics: a counts-only cache row is not a finished one.

Two paths write ``graph_metrics`` under the same primary key, the pipeline run
id: the full computation here, and ``get_datasets_graph_counts``, which fills
only ``num_nodes``/``num_edges`` because that is all the graph-summary
endpoints read. The cache check used to accept *any* row for the run, so once
the cheap path had written one, this function returned it forever and that
run's token count and connectivity metrics read as NULL with nothing reporting
they had never been computed.

Driven against a real SQLite-backed relational engine so the round-trip — the
flag's default, the write, the re-read — is the real one, with only the graph
engine mocked.
"""

import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import GraphMetrics
from cognee.modules.metrics.operations import get_pipeline_run_metrics

# The package re-exports the function under its module's name, which shadows
# the submodule, so the module has to come from sys.modules for patch.object to
# reach its globals.
metrics_module = sys.modules["cognee.modules.metrics.operations.get_pipeline_run_metrics"]

PIPELINE_RUN_ID = uuid4()

FULL_METRICS = {
    "num_nodes": 12,
    "num_edges": 34,
    "mean_degree": 5.6,
    "edge_density": 0.25,
    "num_connected_components": 2,
    "sizes_of_connected_components": [8, 4],
    "num_selfloops": 1,
    "diameter": 3,
    "avg_shortest_path_length": 1.5,
    "avg_clustering": 0.75,
}


async def _make_engine():
    """A throwaway SQLite-backed relational engine with the real schema."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp.name}")
    await engine.create_database()
    return engine


def _graph_engine():
    """A graph engine whose get_graph_metrics call can be asserted on."""
    return AsyncMock(
        return_value=SimpleNamespace(get_graph_metrics=AsyncMock(return_value=FULL_METRICS))
    )


async def _stored_rows(engine):
    async with engine.get_async_session() as session:
        return (
            (await session.execute(select(GraphMetrics).where(GraphMetrics.id == PIPELINE_RUN_ID)))
            .scalars()
            .all()
        )


async def _write_counts_only_row(engine):
    """The row the graph-summary counting path leaves behind: the two counts it
    reads, and nothing else set."""
    async with engine.get_async_session() as session:
        session.add(GraphMetrics(id=PIPELINE_RUN_ID, num_nodes=12, num_edges=34))
        await session.commit()


@pytest.mark.asyncio
async def test_a_row_written_without_the_flag_defaults_to_partial():
    """The column has to default False, or the counting path would have to
    remember to say so and a future writer could forget."""
    engine = await _make_engine()
    await _write_counts_only_row(engine)

    (row,) = await _stored_rows(engine)
    assert row.has_full_metrics is False
    assert row.num_nodes == 12
    assert row.num_tokens is None


@pytest.mark.asyncio
async def test_a_counts_only_row_is_not_a_cache_hit_and_is_completed_in_place():
    """The bug this exists for: the cheap write must not end up answering for
    metrics it never computed."""
    engine = await _make_engine()
    await _write_counts_only_row(engine)
    graph_engine = _graph_engine()

    with (
        patch.object(metrics_module, "get_relational_engine", lambda: engine),
        patch.object(metrics_module, "get_graph_engine", graph_engine),
    ):
        [metrics] = await get_pipeline_run_metrics(
            SimpleNamespace(pipeline_run_id=PIPELINE_RUN_ID), include_optional=True
        )

    # It recomputed rather than serving the partial row.
    graph_engine.return_value.get_graph_metrics.assert_awaited_once_with(True)
    assert metrics.avg_clustering == 0.75

    # Completed in place: still one row under this run id, now flagged whole.
    (row,) = await _stored_rows(engine)
    assert row.has_full_metrics is True
    assert row.diameter == 3
    assert row.mean_degree == 5.6


@pytest.mark.asyncio
async def test_a_completed_row_is_still_served_from_cache():
    """The flag must not turn every call into a recompute."""
    engine = await _make_engine()
    await _write_counts_only_row(engine)

    with (
        patch.object(metrics_module, "get_relational_engine", lambda: engine),
        patch.object(metrics_module, "get_graph_engine", _graph_engine()),
    ):
        await get_pipeline_run_metrics(
            SimpleNamespace(pipeline_run_id=PIPELINE_RUN_ID), include_optional=True
        )

    second_call = _graph_engine()
    with (
        patch.object(metrics_module, "get_relational_engine", lambda: engine),
        patch.object(metrics_module, "get_graph_engine", second_call),
    ):
        [metrics] = await get_pipeline_run_metrics(
            SimpleNamespace(pipeline_run_id=PIPELINE_RUN_ID), include_optional=True
        )

    second_call.return_value.get_graph_metrics.assert_not_awaited()
    assert metrics.avg_clustering == 0.75


@pytest.mark.asyncio
async def test_a_row_computed_without_optional_metrics_stays_a_cache_miss():
    """include_optional=False fills diameter/avg_clustering/etc. with -1
    sentinels rather than computing them, so such a row must not be flagged
    has_full_metrics — otherwise a later include_optional=True caller for the
    same run would serve those sentinels back forever instead of computing
    the real optional metrics."""
    engine = await _make_engine()
    await _write_counts_only_row(engine)

    sentinel_metrics = {**FULL_METRICS, "num_selfloops": -1, "diameter": -1}
    graph_engine = AsyncMock(
        return_value=SimpleNamespace(get_graph_metrics=AsyncMock(return_value=sentinel_metrics))
    )

    with (
        patch.object(metrics_module, "get_relational_engine", lambda: engine),
        patch.object(metrics_module, "get_graph_engine", graph_engine),
    ):
        await get_pipeline_run_metrics(
            SimpleNamespace(pipeline_run_id=PIPELINE_RUN_ID), include_optional=False
        )

    (row,) = await _stored_rows(engine)
    assert row.has_full_metrics is False

    # A later include_optional=True call must recompute rather than trust
    # the sentinel-filled row.
    second_call = _graph_engine()
    with (
        patch.object(metrics_module, "get_relational_engine", lambda: engine),
        patch.object(metrics_module, "get_graph_engine", second_call),
    ):
        [metrics] = await get_pipeline_run_metrics(
            SimpleNamespace(pipeline_run_id=PIPELINE_RUN_ID), include_optional=True
        )

    second_call.return_value.get_graph_metrics.assert_awaited_once_with(True)
    assert metrics.avg_clustering == 0.75
