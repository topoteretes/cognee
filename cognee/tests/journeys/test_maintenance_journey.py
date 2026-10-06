"""Journey: maintenance jobs run after cognify, through the real entry points.

The unit tests pin the runner's rules with fake jobs. These drive real
``remember`` / ``cognify`` / ``improve`` calls on the default stores (LanceDB,
SQLite, Ladybug, multi-user on) with the deterministic mock LLM, and record
every maintenance pass cognify starts: when it ran, for which dataset, against
which vector store, and what the ``vector_compaction`` job reported.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
from dataclasses import dataclass, field
from uuid import UUID

import pytest

import cognee
import cognee.api.v1.cognify.cognify
from cognee.modules.maintenance import JobResult

cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]

DOC_A = (
    "Title: Harrowgate Lighthouse\n\nThe Harrowgate Lighthouse was relit by keeper Odile Brannock "
    "after the storm of the long winter."
)
DOC_B = (
    "Title: Sallowmere Ferry\n\nThe Sallowmere Ferry is skippered by Tomasz Ferrante and crosses "
    "to the lighthouse island twice a day."
)
DOC_C = (
    "Title: Quarry Choir\n\nThe Quarry Choir rehearses in the old slate quarry under conductor "
    "Ines Vauquelin."
)


@dataclass
class Pass:
    pipeline_name: str
    dataset_id: UUID
    last_in_invocation: bool
    run_status: str
    vector_url: str | None
    results: list[JobResult] = field(default_factory=list)

    @property
    def compaction(self) -> JobResult:
        (result,) = [r for r in self.results if r.job == "vector_compaction"]
        return result


async def _cognify_status(dataset_id: UUID) -> str:
    statuses = await cognee.datasets.get_status([dataset_id], ["cognify_pipeline"])
    value = statuses.get(str(dataset_id)) or statuses.get(dataset_id)
    if isinstance(value, dict):
        value = value.get("cognify_pipeline")
    return str(getattr(value, "value", value))


@pytest.fixture
def maintenance_passes(monkeypatch):
    """Every maintenance pass cognify starts, recorded around the real runner."""
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config

    real_run_maintenance = cognify_module.run_maintenance
    passes: list[Pass] = []

    async def recording(**kwargs):
        dataset = kwargs["dataset"]
        record = Pass(
            pipeline_name=kwargs["pipeline_name"],
            dataset_id=dataset.id,
            last_in_invocation=kwargs.get("last_in_invocation", True),
            # Read before the jobs run: the run must already be recorded complete.
            run_status=await _cognify_status(dataset.id),
            vector_url=get_vectordb_context_config().get("vector_db_url"),
        )
        record.results = await real_run_maintenance(**kwargs)
        passes.append(record)
        return record.results

    monkeypatch.setattr(cognify_module, "run_maintenance", recording)
    return passes


def _completed(status: str) -> bool:
    return "COMPLETED" in status.upper()


@pytest.mark.journey
@pytest.mark.asyncio
async def test_cognify_runs_maintenance_once_after_each_completed_run(
    clean_env, default_user, maintenance_passes
):
    """Test 1: remember -> cognify -> one maintenance pass, after completion,
    and vector compaction really merges the fragments the writes left."""
    first = await cognee.remember(DOC_A, dataset_name="journey_maintenance")
    assert first.status == "completed"
    dataset_id = UUID(str(first.dataset_id))
    assert len(maintenance_passes) == 1

    second = await cognee.remember(DOC_B, dataset_name="journey_maintenance")
    assert second.status == "completed"
    assert len(maintenance_passes) == 2, "one maintenance pass per cognify run"

    for record in maintenance_passes:
        assert record.pipeline_name == "cognify_pipeline"
        assert record.dataset_id == dataset_id
        assert _completed(record.run_status), record.run_status
        assert record.compaction.status in ("completed", "already_completed"), record.compaction
        assert record.compaction.counts.get("collection_errors", 0) == 0, record.compaction

    merged = [r.compaction for r in maintenance_passes if r.compaction.status == "completed"]
    assert merged, [r.compaction for r in maintenance_passes]
    assert any(r.counts["fragments_removed"] > r.counts["fragments_added"] for r in merged), merged

    # The data is intact after compaction.
    results = await cognee.search(
        query_text="Who skippers the Sallowmere Ferry?",
        query_type=cognee.SearchType.CHUNKS,
        datasets=["journey_maintenance"],
    )
    assert "ferrante" in str(results).lower()


@pytest.mark.journey
@pytest.mark.asyncio
async def test_background_cognify_runs_maintenance(clean_env, default_user, maintenance_passes):
    """Test 2: cognify(run_in_background=True) still runs the pass, inside the run."""
    from cognee.modules.pipelines.layers.pipeline_execution_mode import (
        _BACKGROUND_PIPELINE_TASKS,
    )

    await cognee.add(DOC_A, dataset_name="journey_maintenance_bg")
    started = await cognee.cognify(["journey_maintenance_bg"], run_in_background=True)
    assert started

    background = list(_BACKGROUND_PIPELINE_TASKS)
    assert background, "the background run was not started"
    await asyncio.gather(*background)

    (record,) = maintenance_passes
    assert _completed(record.run_status), record.run_status
    assert record.compaction.status in ("completed", "already_completed"), record.compaction


@pytest.mark.journey
@pytest.mark.asyncio
async def test_each_dataset_compacts_its_own_store(clean_env, default_user, maintenance_passes):
    """Test 3: multi-user on -- one pass per dataset, each on that dataset's store."""
    await cognee.add(DOC_A, dataset_name="journey_maintenance_one")
    await cognee.add(DOC_B, dataset_name="journey_maintenance_two")

    await cognee.cognify(["journey_maintenance_one", "journey_maintenance_two"])

    assert len(maintenance_passes) == 2
    assert len({record.dataset_id for record in maintenance_passes}) == 2
    urls = [record.vector_url for record in maintenance_passes]
    assert all(urls) and urls[0] != urls[1], urls
    # Only the second dataset is the last of the call, yet with a store per
    # dataset the store-scoped compaction ran for both.
    assert [record.last_in_invocation for record in maintenance_passes] == [False, True]
    for record in maintenance_passes:
        assert record.compaction.status in ("completed", "already_completed"), record.compaction


@pytest.mark.journey
@pytest.mark.asyncio
async def test_maintenance_inside_improve_session_bridging(
    clean_env, default_user, maintenance_passes
):
    """Test 4: improve() bridges a session through a cognify nested inside its own
    run, with the dataset lock already held; the pass must not deadlock on it."""
    seed = await cognee.remember(DOC_A, dataset_name="journey_maintenance_session")
    assert seed.status == "completed"
    before = len(maintenance_passes)

    # A session remember bridges into the graph in the background (auto-improve):
    # that bridge is the improve whose cognify runs nested under the dataset lock.
    stored = await cognee.remember(
        DOC_C, dataset_name="journey_maintenance_session", session_id="maintenance-session"
    )
    assert stored.session_id == "maintenance-session"
    # A test-level guard against a deadlock, not a limit on the product.
    assert await cognee.wait_for_background_tasks(timeout=180), "the session bridge hung"
    result = await cognee.improve(
        "journey_maintenance_session", session_ids=["maintenance-session"]
    )

    stages = {stage.stage: stage for stage in result.stages}
    assert stages["persist_session_qa"].status != "errored", stages["persist_session_qa"]
    bridged = [record for record in maintenance_passes[before:]]
    assert bridged, "the session bridge's cognify ran no maintenance"
    for record in bridged:
        assert record.pipeline_name == "cognify_pipeline"
        assert _completed(record.run_status), record.run_status
        assert record.compaction.status != "errored", record.compaction


@pytest.mark.journey
@pytest.mark.asyncio
async def test_maintenance_finishes_before_engines_are_closed(
    clean_env, default_user, maintenance_passes, monkeypatch
):
    """Test 5: with SUBPROCESS_IDLE_TTL_SECONDS=0 the dataset's subprocess engines
    are closed as soon as its run releases them. The pass runs before that: no
    closed-adapter errors, and no worker process outlives the run."""
    from cognee.infrastructure.databases.dataset_queue import queue as queue_module

    monkeypatch.setenv("SUBPROCESS_IDLE_TTL_SECONDS", "0")
    monkeypatch.setenv("VECTOR_DB_SUBPROCESS_ENABLED", "true")
    previous_queue = queue_module.dataset_queue._instance
    queue_module.dataset_queue._instance = None
    try:
        workers_before = len(multiprocessing.active_children())

        result = await cognee.remember(DOC_B, dataset_name="journey_maintenance_ttl")

        assert result.status == "completed"
        (record,) = maintenance_passes
        assert record.compaction.status in ("completed", "already_completed"), record.compaction
        assert record.compaction.counts.get("collection_errors", 0) == 0, record.compaction

        # Released engines close in the background; give the close a moment.
        for _ in range(50):
            if len(multiprocessing.active_children()) <= workers_before:
                break
            await asyncio.sleep(0.1)
        assert len(multiprocessing.active_children()) <= workers_before, (
            "a worker process outlived its dataset's run"
        )
    finally:
        queue_module.dataset_queue._instance = previous_queue


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_shared_store_is_compacted_once_per_cognify(
    clean_env, default_user, maintenance_passes, monkeypatch
):
    """Multi-user off: every dataset shares one vector store, so compaction runs
    once per multi-dataset cognify, after the last dataset."""
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    await cognee.add(DOC_A, dataset_name="journey_maintenance_shared_one")
    await cognee.add(DOC_B, dataset_name="journey_maintenance_shared_two")

    await cognee.cognify(["journey_maintenance_shared_one", "journey_maintenance_shared_two"])

    assert len(maintenance_passes) == 2
    first, last = maintenance_passes
    assert first.vector_url == last.vector_url, "multi-user off should share one store"
    assert (first.compaction.status, first.compaction.reason) == (
        "skipped",
        "shared_store_runs_after_last_dataset",
    )
    assert last.compaction.status in ("completed", "already_completed"), last.compaction
