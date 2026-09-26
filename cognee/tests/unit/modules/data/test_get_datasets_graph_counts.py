"""get_datasets_graph_counts: the per-cognify-run count cache both graph
summaries read.

The relational reads are patched (they are two plain queries); what this pins
is the decision table around them — never cognified, cache hit, cache miss
computes and caches, an unavailable graph store degrading to zeros instead of
failing the whole batch, and a lost caching race keeping the counts it already
computed. GET /datasets/graph-summary and GET /visualize/brains-summary both
answer from this, so a regression here shows up in two endpoints at once.
"""

import asyncio
import contextlib
import os
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


@pytest.fixture(autouse=True)
def _clean_uncached_cache():
    """The in-process fallback cache is module state; no test may inherit it."""
    counts_module.clear_uncached_counts()
    yield
    counts_module.clear_uncached_counts()


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


# --- the in-process fallback cache -------------------------------------------
#
# Counts that never reached GraphMetrics used to be recounted on every poll.
# The recount is the expensive part: it re-enters the dataset database
# context, which resolves the dataset's own database and takes one of the
# process's few dataset-queue slots. These pin that a failed run is answered
# from memory instead -- and, just as importantly, the cases that must NOT be
# remembered.


@asynccontextmanager
async def _tracking_context(entered, *_args, **_kwargs):
    entered.append(True)
    yield


def _dead_graph_patches(dataset, run_id, entered):
    """A dataset whose latest run has no cached row and whose graph is down."""
    return (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(
            counts_module,
            "set_database_global_context_variables",
            lambda *args, **kwargs: _tracking_context(entered, *args, **kwargs),
        ),
        patch.object(
            counts_module, "get_graph_engine", AsyncMock(side_effect=RuntimeError("graph is down"))
        ),
    )


@pytest.mark.asyncio
async def test_a_failed_count_is_not_recounted_on_the_next_poll():
    """The starvation loop: a broken dataset polled every few seconds must
    not re-enter the database context -- and take a queue slot -- each time."""
    dataset = _dataset()
    run_id = uuid4()
    entered = []

    with contextlib.ExitStack() as stack:
        for context in _dead_graph_patches(dataset, run_id, entered):
            stack.enter_context(context)
        first = await get_datasets_graph_counts([dataset])
        second = await get_datasets_graph_counts([dataset])
        third = await get_datasets_graph_counts([dataset])

    assert len(entered) == 1
    assert first[dataset.id] == DatasetGraphCounts(pipeline_run_id=run_id)
    assert second[dataset.id] == first[dataset.id]
    assert third[dataset.id] == first[dataset.id]


@pytest.mark.asyncio
async def test_correct_counts_whose_write_failed_are_served_rather_than_zeros():
    """The second loop: counting worked, only the cache write failed. The
    repeat poll must answer with those counts, not recount and not zeros."""
    from sqlalchemy.exc import OperationalError

    dataset = _dataset()
    run_id = uuid4()
    entered = []

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(
            counts_module,
            "set_database_global_context_variables",
            lambda *args, **kwargs: _tracking_context(entered, *args, **kwargs),
        ),
        patch.object(counts_module, "get_graph_engine", _graph_engine(num_nodes=7, num_edges=9)),
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
        first = await get_datasets_graph_counts([dataset])
        second = await get_datasets_graph_counts([dataset])

    assert len(entered) == 1
    assert second[dataset.id] == first[dataset.id]
    assert second[dataset.id].num_nodes == 7
    assert second[dataset.id].num_edges == 9


@pytest.mark.asyncio
async def test_a_lost_caching_race_is_not_remembered_in_process():
    """The winner wrote the row, so the durable cache owns the next answer.
    Remembering the loser's copy would pin a stale computed_at=None."""
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
        await get_datasets_graph_counts([dataset])

    assert counts_module._recall_uncached_counts(run_id) is None


@pytest.mark.asyncio
async def test_a_successful_cache_write_is_not_remembered_in_process():
    """The durable row is authoritative; a second copy could only go stale."""
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
        patch.object(counts_module, "get_graph_engine", _graph_engine()),
        patch.object(counts_module, "get_relational_engine", lambda: _fake_engine([])),
    ):
        await get_datasets_graph_counts([dataset])

    assert counts_module._recall_uncached_counts(run_id) is None


@pytest.mark.asyncio
async def test_a_durable_row_still_wins_over_remembered_zeros():
    """Whoever fixes the graph store gets served the moment a row exists --
    the remembered zeros must never shadow it."""
    dataset = _dataset()
    run_id = uuid4()
    entered = []
    cached_at = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)
    row = SimpleNamespace(id=run_id, num_nodes=11, num_edges=12, created_at=cached_at)

    with contextlib.ExitStack() as stack:
        for context in _dead_graph_patches(dataset, run_id, entered):
            stack.enter_context(context)
        await get_datasets_graph_counts([dataset])

    with (
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={run_id: row})),
    ):
        counts = await get_datasets_graph_counts([dataset])

    assert counts[dataset.id] == DatasetGraphCounts(
        pipeline_run_id=run_id, num_nodes=11, num_edges=12, computed_at=cached_at
    )


@pytest.mark.asyncio
async def test_a_new_cognify_run_retries_immediately():
    """Keyed by run id, so re-cognifying is not made to wait out the TTL.

    The repeat poll on the same run id is asserted first, so this test fails
    against a cache keyed by anything coarser AND against one that caches
    nothing at all.
    """
    dataset = _dataset()
    first_run = uuid4()
    entered = []

    with contextlib.ExitStack() as stack:
        for context in _dead_graph_patches(dataset, first_run, entered):
            stack.enter_context(context)
        await get_datasets_graph_counts([dataset])
        await get_datasets_graph_counts([dataset])

    assert len(entered) == 1, "the same run id was recounted"

    with contextlib.ExitStack() as stack:
        for context in _dead_graph_patches(dataset, uuid4(), entered):
            stack.enter_context(context)
        await get_datasets_graph_counts([dataset])

    assert len(entered) == 2


@pytest.mark.asyncio
async def test_an_expired_entry_retries_the_count():
    """A graph store that comes back up is retried once the TTL runs out.

    Control first, so this cannot pass against a cache that does nothing: a
    poll inside the TTL must be suppressed before a poll past it recounts.

    The clock is faked rather than slept through, the same way
    test_llm_payment_required does it -- a real sub-second sleep against a
    real TTL is a margin this suite does not need to carry into CI.
    """
    dataset = _dataset()
    run_id = uuid4()
    entered = []
    clock = [1000.0]

    with (
        patch.dict(os.environ, {"GRAPH_COUNTS_FAILED_TTL_SECONDS": "5"}),
        patch("time.monotonic", lambda: clock[0]),
        contextlib.ExitStack() as stack,
    ):
        for context in _dead_graph_patches(dataset, run_id, entered):
            stack.enter_context(context)
        await get_datasets_graph_counts([dataset])
        await get_datasets_graph_counts([dataset])
        assert len(entered) == 1, "a live entry failed to suppress the recount"

        clock[0] += 6
        await get_datasets_graph_counts([dataset])

    assert len(entered) == 2


@pytest.mark.asyncio
async def test_a_zero_ttl_stops_results_being_held():
    dataset = _dataset()
    run_id = uuid4()
    entered = []

    with (
        patch.dict(os.environ, {"GRAPH_COUNTS_FAILED_TTL_SECONDS": "0"}),
        contextlib.ExitStack() as stack,
    ):
        for context in _dead_graph_patches(dataset, run_id, entered):
            stack.enter_context(context)
        await get_datasets_graph_counts([dataset])
        # Nothing stored, not merely stored-and-instantly-expired: an entry
        # written with expires_at = now + 0 reads as expired on the next poll
        # anyway, so asserting only the recount below would pass with the
        # guard deleted.
        assert counts_module._uncached_counts == {}
        await get_datasets_graph_counts([dataset])

    assert len(entered) == 2


def test_the_cache_stays_bounded():
    """Long-lived processes must not grow an entry per run id forever.

    The peak is what the bound means, not the size at the end: the sweep
    empties the dict wholesale, so a final size well under the cap says
    nothing about whether the cap was ever respected on the way there.
    """
    peak = 0
    for _ in range(counts_module._UNCACHED_COUNTS_MAX_ENTRIES + 50):
        run_id = uuid4()
        counts_module._remember_uncached_counts(
            run_id, DatasetGraphCounts(pipeline_run_id=run_id), ttl=60.0
        )
        peak = max(peak, len(counts_module._uncached_counts))

    assert peak <= counts_module._UNCACHED_COUNTS_MAX_ENTRIES
    assert len(counts_module._uncached_counts) <= counts_module._UNCACHED_COUNTS_MAX_ENTRIES


@pytest.mark.asyncio
async def test_stand_in_zeros_expire_sooner_than_counts_that_were_computed():
    """The two cases the durable cache misses are not equally safe to hold.

    Exact counts that only failed to be written cost nothing to keep; zeros
    standing in for an unreadable graph hide a graph store that came back up,
    so they are held for a much shorter time.
    """
    from sqlalchemy.exc import OperationalError

    broken, unwritable = _dataset(), _dataset()
    broken_run, unwritable_run = uuid4(), uuid4()
    entered = []
    clock = [1000.0]

    def _patches(dataset, run_id, graph_engine):
        return (
            patch.object(
                counts_module,
                "_get_latest_cognify_runs",
                AsyncMock(return_value={dataset.id: _run(dataset.id, run_id)}),
            ),
            patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
            patch.object(
                counts_module,
                "set_database_global_context_variables",
                lambda *args, **kwargs: _tracking_context(entered, *args, **kwargs),
            ),
            patch.object(counts_module, "get_graph_engine", graph_engine),
            patch.object(
                counts_module,
                "get_relational_engine",
                lambda: _fake_engine(
                    [],
                    commit_fails=True,
                    commit_error=OperationalError("COMMIT", {}, Exception("locked")),
                ),
            ),
        )

    dead = AsyncMock(side_effect=RuntimeError("graph is down"))

    with (
        patch.dict(
            os.environ,
            {
                "GRAPH_COUNTS_UNCACHED_TTL_SECONDS": "60",
                "GRAPH_COUNTS_FAILED_TTL_SECONDS": "5",
            },
        ),
        patch("time.monotonic", lambda: clock[0]),
    ):
        with contextlib.ExitStack() as stack:
            for context in _patches(broken, broken_run, dead):
                stack.enter_context(context)
            await get_datasets_graph_counts([broken])

        with contextlib.ExitStack() as stack:
            for context in _patches(unwritable, unwritable_run, _graph_engine(7, 9)):
                stack.enter_context(context)
            await get_datasets_graph_counts([unwritable])

        assert len(entered) == 2

        # Past the failed TTL, inside the uncached one.
        clock[0] += 10

        with contextlib.ExitStack() as stack:
            for context in _patches(broken, broken_run, dead):
                stack.enter_context(context)
            await get_datasets_graph_counts([broken])

        with contextlib.ExitStack() as stack:
            for context in _patches(unwritable, unwritable_run, _graph_engine(7, 9)):
                stack.enter_context(context)
            counts = await get_datasets_graph_counts([unwritable])

    # The zeros were retried; the exact counts were still answered from memory.
    assert len(entered) == 3
    assert counts[unwritable.id].num_nodes == 7


@pytest.mark.asyncio
async def test_two_datasets_in_one_batch_do_not_cross():
    """One remembered, one recounted, in the same call.

    The remembered dataset takes a `continue` in the loop that builds the
    `misses` / `miss_calls` pair recombined by zip(), which is the shape that
    silently pairs one dataset's counts with another's id. Asserted in both
    orderings, since only one of them puts the skip before the append.
    """
    remembered, fresh = _dataset(), _dataset()
    remembered_run, fresh_run = uuid4(), uuid4()
    entered = []

    with contextlib.ExitStack() as stack:
        for context in _dead_graph_patches(remembered, remembered_run, entered):
            stack.enter_context(context)
        await get_datasets_graph_counts([remembered])

    def _mixed_batch(datasets):
        return (
            patch.object(
                counts_module,
                "_get_latest_cognify_runs",
                AsyncMock(
                    return_value={
                        remembered.id: _run(remembered.id, remembered_run),
                        fresh.id: _run(fresh.id, fresh_run),
                    }
                ),
            ),
            patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
            patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
            patch.object(counts_module, "get_graph_engine", _graph_engine(55, 77)),
            patch.object(counts_module, "get_relational_engine", lambda: _fake_engine([])),
        )

    for batch in ([remembered, fresh], [fresh, remembered]):
        with contextlib.ExitStack() as stack:
            for context in _mixed_batch(batch):
                stack.enter_context(context)
            counts = await get_datasets_graph_counts(batch)

        assert counts[remembered.id] == DatasetGraphCounts(pipeline_run_id=remembered_run)
        assert counts[fresh.id].num_nodes == 55
        assert counts[fresh.id].num_edges == 77
