"""get_datasets_graph_counts: the per-cognify-run count cache both graph
summaries read.

The relational reads are patched (they are two plain queries); what this pins
is the decision table around them — never cognified, cache hit, cache miss
computes and caches, an unavailable graph store degrading to zeros instead of
failing the whole batch, and a lost caching race keeping the counts it already
computed. GET /datasets/graph-summary and GET /visualize/brains-summary both
answer from this, so a regression here shows up in two endpoints at once.
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

# The methods package rebinds `get_datasets_graph_counts` to the function, so
# the module itself has to come from sys.modules (same gotcha as the visualize
# API tests) for patch.object to reach its globals.
from cognee.modules.data.methods import get_datasets_graph_counts
from cognee.modules.data.methods.get_datasets_graph_counts import DatasetGraphCounts
from cognee.modules.pipelines.models import PipelineRunStatus

counts_module = sys.modules["cognee.modules.data.methods.get_datasets_graph_counts"]


def _dataset():
    return SimpleNamespace(id=uuid4(), name="billing", owner_id=uuid4())


def _run(dataset_id, pipeline_run_id):
    return SimpleNamespace(
        dataset_id=dataset_id,
        pipeline_run_id=pipeline_run_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
    )


class _FakeSession:
    """Records what would have been cached; optionally fails the commit."""

    def __init__(self, added, commit_fails=False, commit_error=None):
        self._added = added
        self._commit_fails = commit_fails
        self._commit_error = commit_error or IntegrityError(
            "INSERT INTO graph_metrics ...", {}, Exception("unique violation")
        )

    def add(self, instance):
        self._added.append(instance)

    async def commit(self):
        if self._commit_fails:
            raise self._commit_error


def _fake_engine(added, commit_fails=False, commit_error=None):
    @asynccontextmanager
    async def get_async_session():
        yield _FakeSession(added, commit_fails=commit_fails, commit_error=commit_error)

    return SimpleNamespace(get_async_session=get_async_session)


def _graph_engine(num_nodes=12, num_edges=34):
    return AsyncMock(
        return_value=SimpleNamespace(
            get_graph_metrics=AsyncMock(
                return_value={"num_nodes": num_nodes, "num_edges": num_edges}
            )
        )
    )


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_no_datasets_short_circuits_without_querying():
    with patch.object(counts_module, "_get_latest_cognify_runs", AsyncMock()) as latest_runs:
        assert await get_datasets_graph_counts([]) == {}

    latest_runs.assert_not_called()


@pytest.mark.asyncio
async def test_a_never_cognified_dataset_counts_zero_and_names_no_run():
    dataset = _dataset()

    with (
        patch.object(counts_module, "_get_latest_cognify_runs", AsyncMock(return_value={})),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
    ):
        counts = await get_datasets_graph_counts([dataset])

    assert counts == {dataset.id: DatasetGraphCounts()}
    assert counts[dataset.id].pipeline_run_id is None


@pytest.mark.asyncio
async def test_a_cached_run_is_answered_without_touching_the_graph():
    dataset = _dataset()
    run_id = uuid4()
    cached_at = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)
    cached = SimpleNamespace(id=run_id, num_nodes=7, num_edges=9, created_at=cached_at)

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(
            counts_module, "_get_cached_metrics", AsyncMock(return_value={run_id: cached})
        ),
        patch.object(counts_module, "get_graph_engine", AsyncMock()) as graph_engine,
    ):
        counts = await get_datasets_graph_counts([dataset])

    graph_engine.assert_not_called()
    assert counts[dataset.id] == DatasetGraphCounts(
        pipeline_run_id=run_id, num_nodes=7, num_edges=9, computed_at=cached_at
    )


@pytest.mark.asyncio
async def test_a_cache_miss_counts_the_graph_and_caches_it_against_the_run():
    dataset = _dataset()
    run_id = uuid4()
    added = []

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
        patch.object(counts_module, "get_graph_engine", _graph_engine()),
        patch.object(counts_module, "get_relational_engine", lambda: _fake_engine(added)),
    ):
        counts = await get_datasets_graph_counts([dataset])

    assert counts[dataset.id].num_nodes == 12
    assert counts[dataset.id].num_edges == 34
    assert counts[dataset.id].computed_at is not None
    # Cached against the run id, which is what makes the next call free.
    assert [(entry.id, entry.num_nodes, entry.num_edges) for entry in added] == [(run_id, 12, 34)]
    # Flagged partial, so caching counts here cannot make get_pipeline_run_metrics
    # believe that run's token count and connectivity metrics were computed too.
    assert added[0].has_full_metrics is False


@pytest.mark.asyncio
async def test_an_unreadable_graph_degrades_to_zero_without_dropping_the_dataset():
    """One unavailable graph store must not fail, or silently shrink, a batch."""
    readable, unreadable = _dataset(), _dataset()
    readable_run, unreadable_run = uuid4(), uuid4()
    cached = SimpleNamespace(id=readable_run, num_nodes=5, num_edges=6, created_at=None)

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(
                return_value={
                    readable.id: _run(readable.id, readable_run),
                    unreadable.id: _run(unreadable.id, unreadable_run),
                }
            ),
        ),
        patch.object(
            counts_module, "_get_cached_metrics", AsyncMock(return_value={readable_run: cached})
        ),
        patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
        patch.object(
            counts_module, "get_graph_engine", AsyncMock(side_effect=RuntimeError("graph is down"))
        ),
    ):
        counts = await get_datasets_graph_counts([readable, unreadable])

    assert counts[readable.id].num_nodes == 5
    assert counts[unreadable.id] == DatasetGraphCounts(pipeline_run_id=unreadable_run)


@pytest.mark.asyncio
async def test_losing_the_caching_race_still_reports_the_counts_it_computed():
    """A concurrent caller may cache the same run first. The numbers are
    already correct — only computed_at reports that they went uncached."""
    dataset = _dataset()
    run_id = uuid4()

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
        patch.object(counts_module, "get_graph_engine", _graph_engine(num_nodes=3, num_edges=4)),
        patch.object(
            counts_module, "get_relational_engine", lambda: _fake_engine([], commit_fails=True)
        ),
    ):
        counts = await get_datasets_graph_counts([dataset])

    assert counts[dataset.id] == DatasetGraphCounts(
        pipeline_run_id=run_id, num_nodes=3, num_edges=4, computed_at=None
    )


@pytest.mark.asyncio
async def test_an_unexpected_cache_write_failure_degrades_that_dataset_only():
    """A non-IntegrityError commit failure (e.g. a lock/connection error) must
    still degrade to uncached counts for that dataset, not propagate and fail
    the whole batch via asyncio.gather."""
    from sqlalchemy.exc import OperationalError

    dataset_a = _dataset()
    dataset_b = _dataset()
    run_id_a = uuid4()
    run_id_b = uuid4()

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(
                return_value={
                    dataset_a.id: _run(dataset_a.id, run_id_a),
                    dataset_b.id: _run(dataset_b.id, run_id_b),
                }
            ),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
        patch.object(counts_module, "get_graph_engine", _graph_engine(num_nodes=3, num_edges=4)),
        patch.object(
            counts_module,
            "get_relational_engine",
            lambda: _fake_engine(
                [],
                commit_fails=True,
                commit_error=OperationalError("COMMIT", {}, Exception("database is locked")),
            ),
        ),
    ):
        counts = await get_datasets_graph_counts([dataset_a, dataset_b])

    # Both datasets still get their correct, freshly-computed counts — the
    # write failure only cost them the cache, not the response.
    assert counts[dataset_a.id] == DatasetGraphCounts(
        pipeline_run_id=run_id_a, num_nodes=3, num_edges=4, computed_at=None
    )
    assert counts[dataset_b.id] == DatasetGraphCounts(
        pipeline_run_id=run_id_b, num_nodes=3, num_edges=4, computed_at=None
    )


@pytest.mark.asyncio
async def test_missing_metric_keys_read_as_zero_rather_than_none():
    """An adapter that omits a key must not put None into an int field."""
    dataset = _dataset()
    run_id = uuid4()

    empty_metrics = AsyncMock(
        return_value=SimpleNamespace(get_graph_metrics=AsyncMock(return_value={}))
    )

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
        patch.object(counts_module, "get_graph_engine", empty_metrics),
        patch.object(counts_module, "get_relational_engine", lambda: _fake_engine([])),
    ):
        counts = await get_datasets_graph_counts([dataset])

    assert counts[dataset.id].num_nodes == 0
    assert counts[dataset.id].num_edges == 0
