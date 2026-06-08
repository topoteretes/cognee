"""Regression test: ``run_tasks_single`` must preserve ``ctx.data_item``.

Background: the distributed/Modal path builds a ``PipelineContext`` whose
``data_item`` is a single ``Data`` row and then calls
``run_tasks_with_telemetry(data=[data_item], ctx=ctx)`` — wrapping the row in
a list to satisfy first-task signatures that expect a list. Before this
regression test existed, ``run_tasks_single`` pushed ``[head_payload]`` into
``run_worker_pipeline`` with no explicit origin, so the worker's per-call
``dataclasses.replace(shared_ctx, data_item=envelope.origin)`` overwrote
``ctx.data_item`` with the wrapping list. Downstream tasks like
``add_data_points`` then crashed on ``ctx.data_item.id``.

The fix: when the caller has already set ``ctx.data_item``, ``run_tasks_single``
must propagate the original item as ``envelope.origin`` so the per-call ctx
keeps pointing at the unwrapped item.
"""

import asyncio
from dataclasses import dataclass

import pytest

from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.operations.run_tasks_single import run_tasks_single
from cognee.modules.pipelines.tasks.task import Task


@dataclass
class _FakeData:
    """Stand-in for ``cognee.modules.data.models.Data`` — the test only needs
    something with an ``id`` attribute that ``add_data_points`` would read."""

    id: str


@pytest.mark.asyncio
async def test_run_tasks_single_preserves_ctx_data_item_through_stages():
    """Two-stage pipeline where every stage reads ``ctx.data_item.id``. The
    first task receives the wrapping list; downstream stages must still see
    the original ``_FakeData`` in ``ctx.data_item`` rather than the wrap."""
    original = _FakeData(id="row-123")
    seen_ids: list[str] = []

    async def first_task(items, ctx: PipelineContext):
        # Caller wraps the row in a list for this task's signature.
        assert isinstance(items, list) and items[0] is original
        seen_ids.append(ctx.data_item.id)
        return items[0]

    async def second_task(item, ctx: PipelineContext):
        # Without the fix, ``ctx.data_item`` would be ``[original]`` (a list),
        # and ``.id`` would raise AttributeError — the exact regression that
        # surfaced in the Modal CI job.
        seen_ids.append(ctx.data_item.id)
        return item

    ctx = PipelineContext(data_item=original)
    out = []
    async for value in run_tasks_single(
        tasks=[Task(first_task), Task(second_task)],
        data=[original],
        ctx=ctx,
    ):
        out.append(value)

    assert out == [original]
    assert seen_ids == ["row-123", "row-123"]


@pytest.mark.asyncio
async def test_run_tasks_single_without_ctx_data_item_unchanged():
    """When ``ctx.data_item`` is None (or ``ctx`` is None), the per-call ctx
    override falls back to the raw payload — same behavior as before the fix.
    Single-input callers that don't carry an explicit Data row should not be
    forced to provide one."""
    seen = []

    async def stage(value, ctx: PipelineContext):
        seen.append(ctx.data_item)
        return value

    ctx = PipelineContext()  # data_item left as None
    out = []
    async for value in run_tasks_single(
        tasks=[Task(stage)],
        data="payload",
        ctx=ctx,
    ):
        out.append(value)

    assert out == ["payload"]
    # Without an explicit data_item on ctx, origin defaults to the payload —
    # this preserves the pre-fix single-input contract.
    assert seen == ["payload"]


def _run_sync():
    asyncio.run(test_run_tasks_single_preserves_ctx_data_item_through_stages())
    asyncio.run(test_run_tasks_single_without_ctx_data_item_unchanged())


if __name__ == "__main__":
    _run_sync()
