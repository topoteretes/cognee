"""dlt staging is exclusive per pipeline working directory and per staging dataset, and
nothing wider: a connector waiting on its provider holds up only its own dataset."""

import asyncio

import pytest

from cognee.tasks.ingestion import ingest_dlt_source
from cognee.tasks.ingestion.dlt_utils import UNSCOPED_PIPELINE_NAME


async def _second_waits(first: tuple[str, str], second: tuple[str, str]) -> bool:
    """Whether a run on ``second`` (pipeline, dataset) waits while one on ``first`` stages."""
    holding = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with ingest_dlt_source._staging(*first):
            holding.set()
            await release.wait()

    async def stage():
        async with ingest_dlt_source._staging(*second):
            pass

    holder = asyncio.create_task(hold())
    await holding.wait()
    try:
        await asyncio.wait_for(stage(), timeout=0.2)
        return False
    except asyncio.TimeoutError:
        return True
    finally:
        release.set()
        await holder


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first, second, waits",
    [
        # Two channels of one Slack workspace load into one staging database.
        (("ingest_dlt_a", "slack_ws"), ("ingest_dlt_b", "slack_ws"), True),
        # A CSV upload to another dataset goes on while Slack waits on a rate limit.
        (("ingest_dlt_a", "slack_ws"), (UNSCOPED_PIPELINE_NAME, "main_dataset"), False),
        # Unscoped sources share one working directory across datasets.
        ((UNSCOPED_PIPELINE_NAME, "one"), (UNSCOPED_PIPELINE_NAME, "two"), True),
    ],
    ids=["same dataset", "other dataset", "shared unscoped pipeline"],
)
async def test_staging_is_exclusive_per_pipeline_and_dataset(first, second, waits):
    assert await _second_waits(first, second) is waits


@pytest.mark.asyncio
async def test_dropping_a_dataset_waits_for_a_run_on_it():
    """A drop removes the staging database and the pipeline directories of its dataset."""
    assert await _second_waits(("ingest_dlt_a", "slack_ws"), (UNSCOPED_PIPELINE_NAME, "slack_ws"))


@pytest.mark.asyncio
async def test_a_drop_queued_behind_a_run_holds_up_no_csv_ingest_elsewhere():
    """A drop takes the unscoped pipeline every CSV and database source shares. Queued
    behind a Slack run on its dataset, it must not keep that pipeline while it waits
    (second review of #6683)."""
    slack_running = asyncio.Event()
    finish_slack = asyncio.Event()

    async def slack_run():
        async with ingest_dlt_source._staging("ingest_dlt_a", "slack_ws"):
            slack_running.set()
            await finish_slack.wait()

    async def drop():
        async with ingest_dlt_source._staging(UNSCOPED_PIPELINE_NAME, "slack_ws"):
            pass

    running = asyncio.create_task(slack_run())
    await slack_running.wait()
    dropping = asyncio.create_task(drop())
    await asyncio.sleep(0.05)  # the drop is queued on the busy dataset

    async def csv_ingest():
        async with ingest_dlt_source._staging(UNSCOPED_PIPELINE_NAME, "main_dataset"):
            pass

    try:
        await asyncio.wait_for(csv_ingest(), timeout=0.2)
    finally:
        finish_slack.set()
        await asyncio.gather(running, dropping)


# --- The walk pool ---------------------------------------------------------------------


class _Pipeline:
    """A dlt pipeline stand-in whose ``run`` executes ``work`` on the walk thread."""

    def __init__(self, work):
        self.work = work

    def run(self, source, **kwargs):
        return self.work()


@pytest.fixture
def walk_pool():
    ingest_dlt_source.reset_walk_pool()
    yield
    ingest_dlt_source.reset_walk_pool()


@pytest.mark.asyncio
async def test_a_walk_runs_with_its_callers_context(walk_pool):
    """The database context, the current user and the sync reporter are context variables
    the walk reads on its own thread, as ``asyncio.to_thread`` gave them."""
    import contextvars

    current = contextvars.ContextVar("current")
    current.set("the caller's")

    assert await ingest_dlt_source._run_walk(_Pipeline(current.get), None, {}) == "the caller's"


@pytest.mark.asyncio
async def test_a_cancelled_walk_keeps_its_staging_locks_until_its_thread_ends(walk_pool):
    """A thread cannot be stopped: the locks are released only when the walk ends, so no
    second run starts on its pipeline while the first still writes."""
    import threading

    started, finish = threading.Event(), threading.Event()

    def work():
        started.set()
        finish.wait(5)

    async def run():
        async with ingest_dlt_source._staging("ingest_dlt_a", "slack_ws"):
            await ingest_dlt_source._run_walk(_Pipeline(work), None, {})

    task = asyncio.create_task(run())
    await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)

    assert not task.done()
    assert ingest_dlt_source._staging_lock("dataset:slack_ws").locked()

    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ingest_dlt_source._staging_lock("dataset:slack_ws").locked()


@pytest.mark.asyncio
async def test_the_walk_pool_is_as_large_as_the_datasets_worked_on_at_once(walk_pool, monkeypatch):
    monkeypatch.setenv("DATASET_QUEUE_MAX_CONCURRENT", "3")

    assert ingest_dlt_source._walk_executor()._max_workers == 3
