"""dlt walks on different datasets stage side by side (per-dataset staging locks). Each walk
runs in a thread for its whole length, and connectors bridge back to the event loop from it
on every row (``extraction_checkpoint``), where the coroutine may itself need the loop's
default executor (``getaddrinfo`` for a new Postgres connection). Walks must never take every
thread that coroutine needs (review of #6683, second round)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio

import cognee
from cognee.tasks.ingestion import dlt_utils

# More walks than the default executor has threads.
DEFAULT_THREADS = 2
WALKS = 3
# How long one row's bridge to the loop may wait before the walk gives up: far longer than
# the bridge needs when a thread is free, so only starvation reaches it.
BRIDGE_TIMEOUT_SECONDS = 20


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _walk(number: int, loop: asyncio.AbstractEventLoop):
    """A connector walk on its own dataset whose every row asks the loop something that runs
    on the loop's default executor, as a credential check opening a DB session does."""
    import dlt

    async def check_on_the_loop() -> None:
        await asyncio.get_running_loop().run_in_executor(None, lambda: None)

    @dlt.resource(name=f"walk_{number}", primary_key="id")
    def rows():
        for row in range(3):
            asyncio.run_coroutine_threadsafe(check_on_the_loop(), loop).result(
                timeout=BRIDGE_TIMEOUT_SECONDS
            )
            yield {"id": f"{number}-{row}", "title": f"row {row}", "content": f"text {row}"}

    resource = rows()
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, f"walk_{number}")
    return resource


@pytest.mark.asyncio
async def test_more_walks_than_default_executor_threads_all_finish(clean_environment):
    # The databases and the default user, made once instead of raced by the concurrent adds.
    await cognee.add("warm up", dataset_name="warm_up")
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=DEFAULT_THREADS))

    results = await asyncio.gather(
        *(
            cognee.add(
                _walk(number, loop),
                dataset_name=f"walk_dataset_{number}",
                primary_key="id",
                write_disposition="merge",
                max_rows_per_table=0,
            )
            for number in range(WALKS)
        ),
        return_exceptions=True,
    )

    assert [result for result in results if isinstance(result, BaseException)] == []


def _timed_walk(scope: str, events: list, seconds: float):
    """A walk that records when it starts and ends, and takes ``seconds``."""
    import time

    import dlt

    @dlt.resource(name=scope, primary_key="id")
    def rows():
        events.append(("start", scope))
        time.sleep(seconds)
        yield {"id": scope, "title": scope, "content": scope}
        events.append(("end", scope))

    resource = rows()
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, scope)
    return resource


@pytest.mark.asyncio
async def test_walks_queued_on_one_dataset_hold_up_no_other_dataset(clean_environment, monkeypatch):
    """Five CSVs in one add() stage into one dataset, one after another. A walk on another
    dataset, a connector's sync, starts at once instead of behind that queue (swarm F29)."""
    from cognee.tasks.ingestion import ingest_dlt_source

    # The datasets exist before the concurrent adds, which would race to create one.
    await cognee.add("warm up", dataset_name="busy_dataset")
    await cognee.add("warm up", dataset_name="other_dataset")
    # As many walk threads as the busy dataset has walks waiting, and more.
    monkeypatch.setenv("DATASET_QUEUE_MAX_CONCURRENT", "2")
    ingest_dlt_source.reset_walk_pool()
    events: list = []

    def add(scope: str, dataset: str):
        return cognee.add(
            _timed_walk(scope, events, seconds=1.5),
            dataset_name=dataset,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )

    busy = [asyncio.create_task(add(f"busy_{n}", "busy_dataset")) for n in range(5)]
    await asyncio.sleep(0.5)  # the busy walks queue on their dataset
    await add("other", "other_dataset")
    other_started = events.index(("start", "other"))
    first_busy_end = next(i for i, (kind, _) in enumerate(events) if kind == "end")
    await asyncio.gather(*busy)
    ingest_dlt_source.reset_walk_pool()

    assert other_started < first_busy_end or events[first_busy_end] == ("end", "other")
