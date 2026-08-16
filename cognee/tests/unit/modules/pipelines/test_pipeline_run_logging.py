import asyncio
import logging

import cognee
from cognee.modules.pipelines.operations.run_tasks_with_telemetry import run_tasks_with_telemetry
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.infrastructure.databases.relational import create_db_and_tables


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


async def collect_per_item_logs(num_items):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_db_and_tables()
    user = await get_default_user()

    def number_generator(num):
        for i in range(num):
            yield i + 1

    async def add_one(nums):
        for num in nums:
            yield num + 1

    capture = _LogCapture()
    root = logging.getLogger()
    root.addHandler(capture)
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        # run_tasks() invokes run_tasks_with_telemetry() once per data item, so a
        # multi-item run is simulated by invoking it once per item.
        for _ in range(num_items):
            async for _result in run_tasks_with_telemetry(
                [Task(number_generator), Task(add_one)],
                5,
                user,
                "test_pipeline",
            ):
                pass
    finally:
        root.removeHandler(capture)
        root.setLevel(previous_level)

    return capture.messages


def test_per_item_telemetry_does_not_log_pipeline_run_started():
    """Regression for #3724: the per-data-item telemetry wrapper must not emit a
    'Pipeline run started' line, which made a single multi-item cognify() run
    look like the pipeline was starting repeatedly. The run-level start is logged
    once in run_tasks() instead."""
    messages = asyncio.run(collect_per_item_logs(num_items=3))

    started = [message for message in messages if "Pipeline run started" in message]
    processing = [message for message in messages if "Processing data item" in message]

    assert started == [], (
        "run_tasks_with_telemetry must not log 'Pipeline run started' per data item "
        f"(regression for #3724); got {len(started)}"
    )
    assert len(processing) == 3, (
        f"expected one per-item processing log per data item, got {len(processing)}"
    )


if __name__ == "__main__":
    test_per_item_telemetry_does_not_log_pipeline_run_started()
