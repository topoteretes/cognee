"""End-to-end test for background cognify execution.

Runs two full `add -> cognify(run_in_background=True)` cycles on a dataset
(the second cycle adds new data) and verifies, purely from the call return
values and the persisted pipeline status, that every pipeline run reaches a
successful terminal state (DATASET_PROCESSING_COMPLETED, never ERRORED).

Uses the local default databases (SQLite / LanceDB / file-based graph) via
`config.*_root_directory` + prune, and a real LLM (the same credentials CI
provides through secrets). No mocking.
"""

import asyncio
import pathlib
import time

import pytest

import cognee
from cognee.modules.engine.operations.setup import setup as setup_databases
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status


COGNIFY_PIPELINE_NAME = "cognify_pipeline"
COMPLETION_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 3


async def _wait_for_completion(dataset_id) -> PipelineRunStatus:
    """Poll the relational pipeline status until the dataset's cognify run reaches
    a terminal state. Fails on ERRORED or timeout; returns the final status."""
    deadline = time.monotonic() + COMPLETION_TIMEOUT_SECONDS

    while True:
        status_map = await get_pipeline_status([dataset_id], COGNIFY_PIPELINE_NAME)
        status = status_map.get(str(dataset_id))

        assert status != PipelineRunStatus.DATASET_PROCESSING_ERRORED, (
            f"Cognify pipeline for dataset {dataset_id} errored."
        )
        if status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED:
            return status
        if time.monotonic() > deadline:
            raise AssertionError(
                f"Cognify pipeline for dataset {dataset_id} did not complete within "
                f"{COMPLETION_TIMEOUT_SECONDS}s (last status: {status})."
            )

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def _assert_started_ok(started, label: str):
    """Validate the background cognify return value and return the dataset_id."""
    assert isinstance(started, dict) and started, (
        f"{label}: background cognify must return a non-empty "
        f"{{dataset_id: PipelineRunInfo}} map, got {started!r}."
    )
    dataset_id = next(iter(started))
    run_info = started[dataset_id]
    assert not isinstance(run_info, PipelineRunErrored), (
        f"{label}: pipeline returned an errored run: {run_info!r}."
    )
    return dataset_id


@pytest.mark.asyncio
async def test_background_cognify_completes_across_two_add_cycles():
    test_root = pathlib.Path(__file__).parent
    cognee.config.data_root_directory(
        str((test_root / ".data_storage" / "test_background_cognify").resolve())
    )
    cognee.config.system_root_directory(
        str((test_root / ".cognee_system" / "test_background_cognify").resolve())
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup_databases()

    dataset_name = "background_cognify_e2e"

    # ---- Round 1: add data, cognify in the background ----------------------
    await cognee.add(
        "Alan Turing was a British mathematician who is widely considered the "
        "father of theoretical computer science and conceived the Turing machine.",
        dataset_name,
    )
    started_first = await cognee.cognify([dataset_name], run_in_background=True)
    dataset_id = _assert_started_ok(started_first, "Round 1")

    assert await _wait_for_completion(dataset_id) == PipelineRunStatus.DATASET_PROCESSING_COMPLETED

    # ---- Round 2: add NEW data to the same dataset, cognify again ----------
    await cognee.add(
        "Ada Lovelace wrote the first published algorithm intended to be carried "
        "out by Charles Babbage's Analytical Engine.",
        dataset_name,
    )
    started_second = await cognee.cognify([dataset_name], run_in_background=True)
    second_dataset_id = _assert_started_ok(started_second, "Round 2")
    assert second_dataset_id == dataset_id, (
        "Re-cognifying the same dataset must report the same dataset_id."
    )

    assert await _wait_for_completion(dataset_id) == PipelineRunStatus.DATASET_PROCESSING_COMPLETED
