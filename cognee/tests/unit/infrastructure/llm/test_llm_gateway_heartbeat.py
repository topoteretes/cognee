"""Tests for progress reporting at the LLM gateway.

Task boundaries are too coarse to observe a run that is busy inside one long
LLM call. Every structured output call funnels through the gateway, so a
completed call is the finest liveness evidence available. These tests pin the
two properties that matter: the report is attributed to the right run, and it
can never cost a caller their completion.
"""

import asyncio
import importlib
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_pipeline_run_id

# `from cognee.infrastructure.llm import LLMGateway` binds the exported *class*,
# because the package __init__ rebinds the submodule's name. Go through
# importlib for the module that holds the wrapper under test.
gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")


@pytest.fixture(autouse=True)
def _clear_run_context():
    token = current_pipeline_run_id.set(None)
    yield
    current_pipeline_run_id.reset(token)


async def _returns(value):
    return value


@pytest.mark.asyncio
async def test_completed_call_reports_progress_for_the_active_run(monkeypatch):
    reported = []

    async def _spy(pipeline_run_id):
        reported.append(pipeline_run_id)
        return True

    monkeypatch.setattr(
        "cognee.modules.pipelines.operations.heartbeat_pipeline_run.heartbeat_pipeline_run",
        _spy,
    )

    run_id = uuid4()
    current_pipeline_run_id.set(run_id)

    result = await gateway_module._heartbeat_pipeline_run_after(_returns("answer"))

    assert result == "answer"
    assert reported == [run_id]


@pytest.mark.asyncio
async def test_no_active_run_reports_nothing_addressable(monkeypatch):
    """Queries and other non-pipeline LLM traffic have no run to stamp."""
    reported = []

    async def _spy(pipeline_run_id):
        reported.append(pipeline_run_id)
        return False

    monkeypatch.setattr(
        "cognee.modules.pipelines.operations.heartbeat_pipeline_run.heartbeat_pipeline_run",
        _spy,
    )

    assert await gateway_module._heartbeat_pipeline_run_after(_returns(1)) == 1
    assert reported == [None]


@pytest.mark.asyncio
async def test_reporting_failure_never_costs_the_caller_its_result(monkeypatch):
    async def _explode(_pipeline_run_id):
        raise RuntimeError("relational database is gone")

    monkeypatch.setattr(
        "cognee.modules.pipelines.operations.heartbeat_pipeline_run.heartbeat_pipeline_run",
        _explode,
    )

    current_pipeline_run_id.set(uuid4())

    assert await gateway_module._heartbeat_pipeline_run_after(_returns("kept")) == "kept"


@pytest.mark.asyncio
async def test_llm_failure_still_propagates(monkeypatch):
    """The wrapper must not swallow the call it is wrapping."""

    async def _spy(_pipeline_run_id):
        return True

    monkeypatch.setattr(
        "cognee.modules.pipelines.operations.heartbeat_pipeline_run.heartbeat_pipeline_run",
        _spy,
    )

    async def _failing():
        raise ValueError("model refused")

    with pytest.raises(ValueError, match="model refused"):
        await gateway_module._heartbeat_pipeline_run_after(_failing())


@pytest.mark.asyncio
async def test_concurrent_runs_are_attributed_separately():
    """run_tasks sets the run id inside each item's own task, so gathered work
    for different runs must not report against one another."""
    reported = []

    async def _work(run_id):
        current_pipeline_run_id.set(run_id)
        await asyncio.sleep(0)
        reported.append((run_id, current_pipeline_run_id.get()))

    first, second = uuid4(), uuid4()
    await asyncio.gather(
        asyncio.create_task(_work(first)),
        asyncio.create_task(_work(second)),
    )

    assert {seen for _expected, seen in reported} == {first, second}
    assert all(expected == seen for expected, seen in reported)
    # The outer context is untouched by either task.
    assert current_pipeline_run_id.get() is None
