"""The session-keyed rerun (SDK-593) on the improve() orchestrator.

A run that loses the claim to a run holding one of its sessions asks that holder
for one more pass instead of retrying; the holder runs the stages again before
releasing. Dataset-only runs take no part. Everything else is the shared harness.
"""

import asyncio
import importlib

import pytest

from cognee.modules.improve.result import StageResult
from cognee.modules.observability import COGNEE_IMPROVE_STAGES

from .conftest import FakeStage

session_lock = importlib.import_module("cognee.infrastructure.locks.session_lock")


@pytest.fixture(autouse=True)
def _clean_registry():
    session_lock._improving_sessions.clear()
    session_lock._rerun_requested.clear()
    yield
    session_lock._improving_sessions.clear()
    session_lock._rerun_requested.clear()


async def _start_holder(harness, calls, *, session_ids):
    """A background improve blocked inside its first stage until ``gate`` is set."""
    gate = asyncio.Event()

    async def slow_stage(_inputs):
        await gate.wait()
        return StageResult.completed("slow", items=1)

    harness.use_stages(
        [FakeStage("slow", run=slow_stage, calls=calls), FakeStage("after", calls=calls)]
    )
    holder = await harness.improve(session_ids=session_ids, run_in_background=True)
    await asyncio.sleep(0)  # let the task reach the blocked stage
    assert holder.status == "running"
    return holder, gate


@pytest.mark.asyncio
async def test_lock_loser_on_a_session_requests_a_rerun_and_the_holder_runs_one_more_pass(
    harness,
):
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=["chat_1"])

    loser = await harness.improve(session_ids=["chat_1"])

    assert loser.lock_held
    assert loser.rerun_requested is True
    assert loser.status == "skipped"
    assert calls == ["slow"]  # nothing ran for the loser

    gate.set()
    await holder.wait()

    # First pass, then exactly one extra pass over the same stages.
    assert calls == ["slow", "after", "slow", "after"]
    assert [s.stage for s in holder.stages] == ["slow", "after"]
    assert len(holder.rerun_passes) == 1
    assert [s.stage for s in holder.rerun_passes[0]] == ["slow", "after"]
    assert holder.status == "completed"
    assert holder.stage_summary() == "slow=completed,after=completed,rerun_passes=1"
    # The claim is free again afterwards.
    keys = session_lock.improve_lock_keys(["chat_1"], harness.dataset.id, harness.user.id)
    assert await session_lock.try_acquire_improve_lock_many(keys)


@pytest.mark.asyncio
async def test_dataset_only_collision_makes_no_rerun_request_and_no_extra_pass(harness):
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=[])

    loser = await harness.improve()  # dataset-only, same dataset

    assert loser.lock_held
    assert loser.rerun_requested is False

    gate.set()
    await holder.wait()

    assert calls == ["slow", "after"]
    assert holder.rerun_passes == []
    assert holder.stage_summary() == "slow=completed,after=completed"


@pytest.mark.asyncio
async def test_session_loser_against_a_dataset_only_holder_gets_no_promise(harness):
    """The holder owns no session key, so there is nobody to ask; the loser is told so."""
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=[])

    loser = await harness.improve(session_ids=["chat_1"])

    assert loser.lock_held
    assert loser.rerun_requested is False

    gate.set()
    await holder.wait()
    assert holder.rerun_passes == []


@pytest.mark.asyncio
async def test_rerun_passes_are_bounded(harness):
    """A stage that keeps re-requesting (standing in for a stream of losers) cannot
    keep the holder alive forever."""
    calls = []
    improve_mod = harness.improve_mod
    session_key = f"session:{harness.user.id}:chat_1"

    async def nagging_stage(_inputs):
        calls.append("nag")
        await session_lock.request_improve_rerun_many([session_key])
        return StageResult.completed("nag", items=1)

    harness.use_stages([FakeStage("nag", run=nagging_stage)])

    result = await harness.improve(session_ids=["chat_1"])

    assert len(calls) == improve_mod.IMPROVE_MAX_RERUN_PASSES
    assert len(result.rerun_passes) == improve_mod.IMPROVE_MAX_RERUN_PASSES - 1
    # The lock is released at the bound; the leftover request is the next claimant's.
    keys = session_lock.improve_lock_keys(["chat_1"], harness.dataset.id, harness.user.id)
    assert await session_lock.try_acquire_improve_lock_many(keys)
    assert session_key not in session_lock._rerun_requested


@pytest.mark.asyncio
async def test_an_errored_stage_in_a_rerun_pass_marks_the_run_errored(harness):
    calls = []
    session_key = f"session:{harness.user.id}:chat_1"

    async def flaky(_inputs):
        calls.append("flaky")
        if len(calls) == 1:
            await session_lock.request_improve_rerun_many([session_key])
            return StageResult.completed("flaky", items=1)
        return StageResult.errored("flaky", RuntimeError("second pass boom"))

    harness.use_stages([FakeStage("flaky", run=flaky)])

    result = await harness.improve(session_ids=["chat_1"])

    assert calls == ["flaky", "flaky"]
    assert result.stages[0].status == "completed"
    assert result.rerun_passes[0][0].status == "errored"
    assert result.status == "errored"
    from cognee.modules.pipelines.models import OperationOutcome

    assert harness.operations[-1].outcome == OperationOutcome.FAILED
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == "flaky=completed,rerun_passes=1"


@pytest.mark.asyncio
async def test_rerun_pass_refreshes_the_enrichment_stamp_without_a_clobber_warning(harness, caplog):
    session_key = f"session:{harness.user.id}:chat_1"
    passes = {"n": 0}

    async def stamping(_inputs):
        passes["n"] += 1
        if passes["n"] == 1:
            await session_lock.request_improve_rerun_many([session_key])
        result = StageResult.completed("stamping", items=1)
        result._run_info_stamp = {"stamping": {"pass": passes["n"]}}
        return result

    harness.use_stages([FakeStage("stamping", run=stamping)])

    with caplog.at_level("WARNING", logger="record_operation"):
        result = await harness.improve(session_ids=["chat_1"])

    assert len(result.rerun_passes) == 1
    assert harness.operations[-1].run_info == {"stamping": {"pass": 2}}
    assert "run_info keys overwritten" not in caplog.text


@pytest.mark.asyncio
async def test_lock_held_result_serializes_the_rerun_fields(harness):
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=["chat_1"])
    loser = await harness.improve(session_ids=["chat_1"])
    gate.set()
    await holder.wait()

    loser_body = loser.model_dump(mode="json")
    holder_body = holder.model_dump(mode="json")
    assert loser_body["rerun_requested"] is True
    assert loser_body["rerun_passes"] == []
    assert holder_body["rerun_requested"] is False
    assert len(holder_body["rerun_passes"]) == 1
    assert [s["stage"] for s in holder_body["rerun_passes"][0]] == ["slow", "after"]
