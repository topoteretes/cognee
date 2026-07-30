"""Test that PipelineContext.extras flows from caller through to task functions.

Validates that the fix for dexters1's comment #26 actually works:
custom pipelines can pass arbitrary state via extras and read it inside tasks.
"""

from dataclasses import dataclass
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base


@dataclass
class _FakeUser:
    """Minimal user stub — run_tasks_base needs user.id for telemetry."""

    id: str = str(uuid4())
    tenant_id: str = None


_USER = _FakeUser()

# -- A custom task that reads extras from the pipeline context ----------------


async def enrich_with_custom_config(data, ctx: PipelineContext = None):
    """A task that uses ctx.extras to get caller-supplied configuration."""
    multiplier = ctx.extras.get("score_multiplier", 1) if ctx else 1
    prefix = ctx.extras.get("label_prefix", "") if ctx else ""

    if isinstance(data, list):
        return [{"value": item * multiplier, "label": f"{prefix}{item}"} for item in data]
    return {"value": data * multiplier, "label": f"{prefix}{data}"}


async def filter_by_threshold(data, ctx: PipelineContext = None):
    """A second task in the chain that also reads extras."""
    threshold = ctx.extras.get("min_threshold", 0) if ctx else 0

    if isinstance(data, list):
        return [item for item in data if item.get("value", 0) >= threshold]
    return data if data.get("value", 0) >= threshold else None


# -- Tests --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extras_flow_through_single_task():
    """Extras are available inside a task that declares ctx: PipelineContext."""
    ctx = PipelineContext(
        pipeline_name="test_pipeline",
        extras={"score_multiplier": 10, "label_prefix": "item_"},
    )

    tasks = [Task(enrich_with_custom_config)]
    results = []
    async for result in run_tasks_base(tasks, [1, 2, 3], user=_USER, ctx=ctx):
        results.append(result)

    assert len(results) > 0
    flat = results[0] if isinstance(results[0], list) else results
    assert flat[0]["value"] == 10  # 1 * 10
    assert flat[0]["label"] == "item_1"
    assert flat[1]["value"] == 20  # 2 * 10
    assert flat[2]["value"] == 30  # 3 * 10


@pytest.mark.asyncio
async def test_extras_flow_through_chained_tasks():
    """Extras persist across a two-task pipeline chain."""
    ctx = PipelineContext(
        pipeline_name="test_chain",
        extras={"score_multiplier": 5, "label_prefix": "x", "min_threshold": 12},
    )

    tasks = [
        Task(enrich_with_custom_config),
        Task(filter_by_threshold),
    ]
    results = []
    async for result in run_tasks_base(tasks, [1, 2, 3], user=_USER, ctx=ctx):
        results.append(result)

    # After enrichment: values are 5, 10, 15
    # After filter (threshold=12): only value=15 survives
    flat = results[0] if isinstance(results[0], list) else results
    assert len(flat) == 1
    assert flat[0]["value"] == 15
    assert flat[0]["label"] == "x3"


@pytest.mark.asyncio
async def test_extras_default_empty_dict():
    """When no extras are provided, ctx.extras is an empty dict (not None)."""
    ctx = PipelineContext(pipeline_name="test_default")
    assert ctx.extras == {}
    assert isinstance(ctx.extras, dict)


@pytest.mark.asyncio
async def test_task_without_ctx_ignores_extras():
    """A task that doesn't declare ctx still works — extras are just not injected."""

    async def plain_task(data):
        return [x * 2 for x in data]

    ctx = PipelineContext(
        pipeline_name="test_no_ctx",
        extras={"should_be_ignored": True},
    )

    tasks = [Task(plain_task)]
    results = []
    async for result in run_tasks_base(tasks, [1, 2, 3], user=_USER, ctx=ctx):
        results.append(result)

    flat = results[0] if isinstance(results[0], list) else results
    assert flat == [2, 4, 6]


@pytest.mark.asyncio
async def test_run_tasks_copies_extras_per_item(monkeypatch, runner_plumbing):
    """Each data item gets its OWN copy of caller-supplied extras.

    A shared dict would let one item's ctx.extras mutations (e.g. DLT dedup
    sets) leak into every concurrently running item — the same bug class as
    the shared Task-kwarg sets, one layer up.
    """
    from types import SimpleNamespace
    from uuid import uuid4

    import cognee.modules.pipelines.operations.run_tasks as run_tasks_module

    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)

    captured_ctxs = []

    async def _fake_item_run(data_item, ds, item_tasks, name, pid, rid, ctx, *args):
        captured_ctxs.append(ctx)
        ctx.extras["mutated_by"] = str(data_item)
        return {"run_info": "ok"}

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _fake_item_run)

    caller_extras = {"score_multiplier": 3}
    async for _ in run_tasks_module.run_tasks.__wrapped__(
        tasks="TASKS",
        dataset_id=dataset.id,
        data=["item_a", "item_b"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="custom_pipeline",
        extras=caller_extras,
    ):
        pass

    assert len(captured_ctxs) == 2
    # Distinct dict objects, each seeded with the caller's values.
    assert captured_ctxs[0].extras is not captured_ctxs[1].extras
    assert all(ctx.extras["score_multiplier"] == 3 for ctx in captured_ctxs)
    # Per-item mutations stayed per-item and never reached the caller's dict.
    assert captured_ctxs[0].extras["mutated_by"] != captured_ctxs[1].extras["mutated_by"]
    assert caller_extras == {"score_multiplier": 3}
