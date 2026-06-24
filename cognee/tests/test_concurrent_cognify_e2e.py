"""End-to-end test: two background cognify runs on one dataset, each isolated to its own file.

Flow (exactly as a caller would do it):
    add(file_one)   -> cognify(run_in_background=True)   # run #1 starts, returns immediately
    add(file_two)   -> cognify(run_in_background=True)   # run #2 starts, returns immediately
    gather both runs to completion

The per-dataset asyncio lock in `run_pipeline_per_dataset` serializes everything
touching the dataset, so the two runs cannot clobber each other. We assert, purely
from each run's return value (its PipelineRunCompleted.data_ingestion_info):

    * both runs complete successfully (no errored run),
    * they are two distinct pipeline runs, and
    * run #1 cognified ONLY file_one and run #2 cognified ONLY file_two
      (incremental loading skips file_one on the second run).

The two files are intentionally a bit larger than a single sentence so cognify #1
is still processing in the background when we add the second file and start
cognify #2 — i.e. we actually exercise the lock's contention window.

Uses the local default databases (SQLite / LanceDB / file-based graph) via
`config.*_root_directory` + prune, and a real LLM (the credentials CI provides
through secrets). No mocking.
"""

import asyncio
import pathlib

import pytest

import cognee
from cognee.modules.engine.operations.setup import setup as setup_databases
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)
from cognee.modules.pipelines.queues.pipeline_run_info_queues import get_from_queue


POLL_INTERVAL_SECONDS = 1


def _make_document(subject: str, paragraphs: int = 40) -> str:
    """Build a distinct, multi-paragraph document so cognify produces several chunks."""
    return "\n\n".join(
        f"{subject} note {i + 1}: In context {i + 1}, {subject} is associated with a "
        f"distinct development number {i + 1}. This particular detail about {subject} "
        f"is documented as item {i + 1} and connects to topic {i + 1} in measurable ways."
        for i in range(paragraphs)
    )


async def _await_run_completion(pipeline_run_id) -> PipelineRunCompleted:
    """Drain a background run's info queue until it reaches a terminal status.

    No timeout on purpose: real LLM calls are slow and not always stable, so a
    deadline here would only produce flaky failures. We wait for the run.
    """
    while True:
        info = get_from_queue(pipeline_run_id)
        if info is None:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue
        if isinstance(info, PipelineRunErrored):
            raise AssertionError(f"Run {pipeline_run_id} errored: {info!r}.")
        if isinstance(info, PipelineRunCompleted):
            return info
        # Non-terminal yields are ignored; keep draining.


def _started_run_id(started, label: str):
    """Extract the pipeline_run_id from a background cognify's return value."""
    assert isinstance(started, dict) and started, (
        f"{label}: background cognify must return a non-empty "
        f"{{dataset_id: PipelineRunStarted}} map, got {started!r}."
    )
    return next(iter(started.values())).pipeline_run_id


def _processed_data_ids(completed: PipelineRunCompleted) -> set:
    """Data ids the run actually cognified (PipelineRunCompleted items, not skipped ones)."""
    processed = set()
    for item in completed.data_ingestion_info or []:
        if isinstance(item.get("run_info"), PipelineRunCompleted):
            processed.add(str(item["data_id"]))
    return processed


@pytest.mark.asyncio
async def test_two_background_cognify_runs_are_isolated_per_file():
    test_root = pathlib.Path(__file__).parent
    cognee.config.data_root_directory(
        str((test_root / ".data_storage" / "test_concurrent_cognify").resolve())
    )
    cognee.config.system_root_directory(
        str((test_root / ".cognee_system" / "test_concurrent_cognify").resolve())
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup_databases()

    dataset_name = "concurrent_cognify_e2e"

    # ---- First file: add, then cognify in the background -------------------
    add_one = await cognee.add(_make_document("Alan Turing"), dataset_name)
    data_id_one = str(add_one.data_ingestion_info[0]["data_id"])
    started_one = await cognee.cognify([dataset_name], run_in_background=True)
    run_id_one = _started_run_id(started_one, "cognify #1")

    # ---- Second file: add NEW file, then a second background cognify -------
    # (cognify #1 is still processing under the per-dataset lock here.)
    add_two = await cognee.add(_make_document("Ada Lovelace"), dataset_name)
    data_id_two = str(add_two.data_ingestion_info[0]["data_id"])
    started_two = await cognee.cognify([dataset_name], run_in_background=True)
    run_id_two = _started_run_id(started_two, "cognify #2")

    assert data_id_one != data_id_two, "The two adds must produce two distinct files."
    assert run_id_one != run_id_two, "Each cognify must be its own pipeline run."

    # ---- Gather both background runs to completion -------------------------
    completed_one, completed_two = await asyncio.gather(
        _await_run_completion(run_id_one),
        _await_run_completion(run_id_two),
    )

    # ---- Each run cognified exactly its own file --------------------------
    assert _processed_data_ids(completed_one) == {data_id_one}, (
        "cognify #1 must have processed only the first file."
    )
    assert _processed_data_ids(completed_two) == {data_id_two}, (
        "cognify #2 must have processed only the second file."
    )
