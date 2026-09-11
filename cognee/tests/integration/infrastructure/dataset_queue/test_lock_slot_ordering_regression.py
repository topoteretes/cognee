"""Regression test for SDK-483: DatasetQueue slot / dataset-lock order inversion.

Races an improve-type pipeline (``apply_feedback_weights_pipeline``) against a
pipeline run on the same dataset with ``DATASET_QUEUE_MAX_CONCURRENT=1``.
Before the fix, the improve path held its queue slot while waiting on the
dataset lock, so the interleave below deadlocked permanently:

    improve: [slot] ------------------> wait lock(D)   (holds the only slot)
    run:            lock(D) -> STARTED -> wait slot

The scenario runs in a subprocess: the queue reads its env config at first
use and both primitives are process-global singletons, so in-process pytest
state could mask or leak the condition. No LLM call is made on either path.
"""

import os
import subprocess
import sys

import pytest

_SCENARIO = """
import asyncio
import os
import sys
import tempfile

workdir = tempfile.mkdtemp(prefix="sdk483_regression_")
os.environ["DATA_ROOT_DIRECTORY"] = os.path.join(workdir, "data")
os.environ["SYSTEM_ROOT_DIRECTORY"] = os.path.join(workdir, "system")
os.environ["DATASET_QUEUE_MAX_CONCURRENT"] = "1"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "true"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["TELEMETRY_DISABLED"] = "1"
os.environ.setdefault("LLM_API_KEY", "sk-dummy-never-called")

import cognee
from cognee.memify_pipelines import apply_feedback_weights as afw_module
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user

DATASET = "sdk483_dataset"

_real_memify = afw_module.memify


async def delayed_memify(**kwargs):
    # Widen the historical race window between the improve pipeline's start and
    # its dataset-lock request, standing in for the migration gate and auth DB
    # reads. With the ordering fixed no slot is held here, so the run pipeline
    # below proceeds and both sides complete.
    await asyncio.sleep(2)
    return await _real_memify(**kwargs)


async def passthrough(data):
    return data


async def run_locked_pipeline(user):
    # The exact locking machinery cognify runs (dataset lock -> STARTED ->
    # queue slot), with a trivial task so no LLM is involved.
    async for _ in run_pipeline(
        tasks=[Task(passthrough)],
        data=["item"],
        datasets=[DATASET],
        user=user,
        pipeline_name="cognify_pipeline",
    ):
        pass


async def main():
    await cognee.add("regression seed", dataset_name=DATASET)
    user = await get_default_user()
    afw_module.memify = delayed_memify

    improve_task = asyncio.create_task(
        afw_module.apply_feedback_weights_pipeline(
            user=user, session_ids=["s1"], dataset=DATASET
        )
    )
    await asyncio.sleep(1)
    run_task = asyncio.create_task(run_locked_pipeline(user))

    done, pending = await asyncio.wait({improve_task, run_task}, timeout=90)
    if pending:
        for task in pending:
            task.cancel()
        print("DEADLOCK: tasks still pending after 90s", file=sys.stderr)
        sys.exit(1)

    # Completion (even with a per-task error) is the pass criterion: the
    # regression is the hang, not the pipelines' business outcome.
    for task in done:
        error = task.exception()
        if error is not None:
            print(f"task finished with {error!r} (no hang - acceptable)")
    print("COMPLETED")


asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_improve_and_pipeline_run_on_same_dataset_complete():
    """Concurrent improve-type + locked pipeline run must complete, not hang."""
    result = subprocess.run(
        [sys.executable, "-c", _SCENARIO],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, (
        f"scenario failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    assert "COMPLETED" in result.stdout
