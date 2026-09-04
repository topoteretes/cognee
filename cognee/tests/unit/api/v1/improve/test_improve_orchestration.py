"""Orchestration tests for improve(): order, gates, lock, fatal/errored, background.

All stages are fakes; nothing below the orchestrator runs.
"""

import asyncio
import importlib
from uuid import uuid4

import pytest

from cognee.modules.improve import (
    REASON_ABORTED_BY_FATAL_STAGE,
    REASON_DISABLED_BY_CONFIG,
    REASON_LOCK_HELD,
    REASON_NO_SESSION_IDS,
    ImproveResult,
)
from cognee.modules.improve.result import StageResult
from cognee.modules.observability import COGNEE_IMPROVE_STAGES
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)

from .conftest import FakeStage

session_lock = importlib.import_module("cognee.infrastructure.locks.session_lock")


def _run_info(cls, dataset_id, **extra):
    return cls(pipeline_run_id=uuid4(), dataset_id=dataset_id, dataset_name="docs", **extra)


@pytest.mark.asyncio
async def test_stages_run_in_registry_order_and_result_lists_each(harness):
    calls = []
    stages = harness.use_stages(
        [FakeStage(f"s{i}", calls=calls) for i in range(1, 5)],
    )

    result = await harness.improve()

    assert isinstance(result, ImproveResult)
    assert calls == ["s1", "s2", "s3", "s4"]
    assert [s.stage for s in result.stages] == [stage.name for stage in stages]
    assert all(s.status == "completed" for s in result.stages)
    assert result.status == "completed"
    assert result.finished is True
    assert result.dataset_id == harness.dataset.id
    assert result.dataset_name == "docs"
    # Every stage saw the same frozen inputs, carrying the resolved id, never the name.
    inputs = stages[0].seen_inputs[0]
    assert inputs.dataset_id == harness.dataset.id
    assert inputs.dataset is harness.dataset
    assert not hasattr(inputs, "run_in_background")
    assert harness.resolve_calls == ["docs"]  # resolved exactly once


@pytest.mark.asyncio
async def test_gate_reasons(harness):
    calls = []
    harness.set_config(stages_disabled=["disabled_one"])
    stages = harness.use_stages(
        [
            FakeStage("needs_sessions", kind="session", calls=calls),
            FakeStage("disabled_one", calls=calls),
            FakeStage("own_gate", gate_reason="opt_in_disabled", calls=calls),
            FakeStage("runs", calls=calls),
        ]
    )

    result = await harness.improve()  # no session_ids

    by_name = {s.stage: s for s in result.stages}
    assert by_name["needs_sessions"].status == "skipped"
    assert by_name["needs_sessions"].reason == REASON_NO_SESSION_IDS
    assert by_name["disabled_one"].status == "skipped"
    assert by_name["disabled_one"].reason == REASON_DISABLED_BY_CONFIG
    assert by_name["own_gate"].status == "skipped"
    assert by_name["own_gate"].reason == "opt_in_disabled"
    assert by_name["runs"].status == "completed"
    assert calls == ["runs"]
    # Run-level gates short-circuit before the stage's own gate is consulted.
    assert stages[0].gate_calls == 0
    assert stages[1].gate_calls == 0
    assert stages[2].gate_calls == 1


@pytest.mark.asyncio
async def test_session_stage_runs_when_session_ids_given(harness):
    calls = []
    harness.use_stages([FakeStage("needs_sessions", kind="session", calls=calls)])

    result = await harness.improve(session_ids=["chat_1", "chat_2"])

    assert calls == ["needs_sessions"]
    assert result.session_ids == ["chat_1", "chat_2"]
    assert result.stages[0].status == "completed"


@pytest.mark.asyncio
async def test_lock_held_returns_every_stage_skipped_never_empty_dict(harness):
    calls = []
    harness.use_stages([FakeStage("a", calls=calls), FakeStage("b", calls=calls)])
    key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([key])
    try:
        result = await harness.improve()
    finally:
        await session_lock.release_improve_lock_many([key])

    assert isinstance(result, ImproveResult)
    assert result.lock_held is True
    assert result.status == "skipped"
    assert [(s.stage, s.status, s.reason) for s in result.stages] == [
        ("a", "skipped", REASON_LOCK_HELD),
        ("b", "skipped", REASON_LOCK_HELD),
    ]
    assert calls == []
    assert result.to_legacy_dict() == {}
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == "a=skipped,b=skipped"


@pytest.mark.asyncio
async def test_lock_is_keyed_to_session_ids_when_given(harness):
    calls = []
    harness.use_stages([FakeStage("a", calls=calls)])
    # Another run holding one of our sessions blocks us; the dataset key does not.
    assert await session_lock.try_acquire_improve_lock_many(["chat_2"])
    try:
        blocked = await harness.improve(session_ids=["chat_1", "chat_2"])
    finally:
        await session_lock.release_improve_lock_many(["chat_2"])
    assert blocked.lock_held

    assert await session_lock.try_acquire_improve_lock_many([f"dataset:{harness.dataset.id}"])
    try:
        allowed = await harness.improve(session_ids=["chat_1", "chat_2"])
    finally:
        await session_lock.release_improve_lock_many([f"dataset:{harness.dataset.id}"])
    assert not allowed.lock_held
    assert calls == ["a"]
    # And the claim is released afterwards.
    assert await session_lock.try_acquire_improve_lock_many(["chat_1", "chat_2"])
    await session_lock.release_improve_lock_many(["chat_1", "chat_2"])


@pytest.mark.asyncio
async def test_fatal_stage_stops_chain_and_raises_with_partial_result(harness):
    calls = []
    boom = RuntimeError("persist failed")
    harness.use_stages(
        [
            FakeStage("first", calls=calls),
            FakeStage("fatal_one", fatal=True, run=lambda _i: boom, calls=calls),
            FakeStage("after_fatal", calls=calls),
            FakeStage("last", calls=calls),
        ]
    )

    with pytest.raises(RuntimeError) as excinfo:
        await harness.improve()

    assert calls == ["first", "fatal_one"]
    partial = excinfo.value.improve_result
    assert isinstance(partial, ImproveResult)
    assert [(s.stage, s.status) for s in partial.stages] == [
        ("first", "completed"),
        ("fatal_one", "errored"),
        ("after_fatal", "skipped"),
        ("last", "skipped"),
    ]
    assert partial.stages[1].error == "RuntimeError: persist failed"
    assert partial.stages[2].reason == REASON_ABORTED_BY_FATAL_STAGE
    assert partial.status == "errored"
    # The lock was released on the way out.
    key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([key])
    await session_lock.release_improve_lock_many([key])
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == (
        "first=completed,fatal_one=errored,after_fatal=skipped,last=skipped"
    )


@pytest.mark.asyncio
async def test_fatal_stage_reporting_errored_run_info_also_stops_chain(harness):
    calls = []
    errored = _run_info(
        PipelineRunErrored, harness.dataset.id, error_class="X", error_message="lost"
    )
    harness.use_stages(
        [
            FakeStage(
                "fatal_one",
                fatal=True,
                run=lambda _i: StageResult.from_pipeline_run("fatal_one", {"d": errored}),
                calls=calls,
            ),
            FakeStage("after", calls=calls),
        ]
    )

    with pytest.raises(Exception) as excinfo:
        await harness.improve()

    partial = excinfo.value.improve_result
    assert calls == ["fatal_one"]
    assert partial.stages[0].status == "errored"
    assert partial.stages[0].error == "lost"
    assert partial.stages[1].reason == REASON_ABORTED_BY_FATAL_STAGE


@pytest.mark.asyncio
async def test_errored_non_fatal_stage_records_and_continues(harness):
    calls = []
    harness.use_stages(
        [
            FakeStage("flaky", run=lambda _i: ValueError("nope"), calls=calls),
            FakeStage("next", calls=calls),
        ]
    )

    result = await harness.improve()

    assert calls == ["flaky", "next"]
    assert result.stages[0].status == "errored"
    assert result.stages[0].error == "ValueError: nope"
    assert result.stages[1].status == "completed"
    assert result.status == "errored"
    assert [s.stage for s in result.ok] == ["next"]


@pytest.mark.asyncio
async def test_stage_status_is_derived_from_pipeline_run_info(harness):
    ds = harness.dataset.id
    completed = _run_info(PipelineRunCompleted, ds)
    already = _run_info(PipelineRunAlreadyCompleted, ds)
    errored = _run_info(PipelineRunErrored, ds, error_message="bad")
    memify_return = {ds: completed}
    harness.use_stages(
        [
            FakeStage(
                "triplet_enrichment",
                run=lambda _i: StageResult.from_pipeline_run("triplet_enrichment", memify_return),
            ),
            FakeStage("b", run=lambda _i: StageResult.from_pipeline_run("b", {ds: already})),
            FakeStage("c", run=lambda _i: StageResult.from_pipeline_run("c", errored)),
        ]
    )

    result = await harness.improve()

    assert [s.status for s in result.stages] == ["completed", "already_completed", "errored"]
    assert result.stages[0].run is completed
    assert result.stages[2].error == "bad"
    # Legacy return shape stays reachable, nested (D4).
    assert result.memify_run is memify_return
    assert result.to_legacy_dict() is memify_return
    assert result.stages[0].duration_ms >= 0


@pytest.mark.asyncio
async def test_background_mode_runs_whole_chain_under_one_lock(harness):
    calls = []
    release = asyncio.Event()

    async def slow_stage(_inputs):
        await release.wait()
        return StageResult.completed("slow", items=1)

    harness.use_stages(
        [
            FakeStage("slow", run=slow_stage, calls=calls),
            FakeStage("after_slow", calls=calls),
        ]
    )
    key = f"dataset:{harness.dataset.id}"

    result = await harness.improve(run_in_background=True)

    assert result.status == "running"
    assert result.background is True
    assert result.stages == []
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == "background"
    # The whole chain, not one stage, is what holds the claim.
    await asyncio.sleep(0)
    assert not await session_lock.try_acquire_improve_lock_many([key])
    assert calls == ["slow"]
    assert "after_slow" not in calls

    release.set()
    finished = await result.wait()

    assert finished is result
    assert result.status == "completed"
    assert calls == ["slow", "after_slow"]
    assert [s.stage for s in result.stages] == ["slow", "after_slow"]
    assert await session_lock.try_acquire_improve_lock_many([key])
    await session_lock.release_improve_lock_many([key])
    # No stage was told about background mode.
    assert not hasattr(harness.improve_mod.DEFAULT_STAGES[0].seen_inputs[0], "run_in_background")


@pytest.mark.asyncio
async def test_background_fatal_error_is_recorded_not_raised(harness):
    harness.use_stages(
        [
            FakeStage("fatal_one", fatal=True, run=lambda _i: RuntimeError("boom")),
            FakeStage(
                "after",
            ),
        ]
    )

    result = await harness.improve(run_in_background=True)
    await result.wait()

    assert result.status == "errored"
    assert result.error == "RuntimeError: boom"
    assert [s.status for s in result.stages] == ["errored", "skipped"]
    key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([key])
    await session_lock.release_improve_lock_many([key])


@pytest.mark.asyncio
async def test_telemetry_hashes_session_ids(harness):
    harness.use_stages([FakeStage("a")])

    await harness.improve(session_ids=["very-secret-session"])

    props = harness.telemetry[0]["properties"]
    assert props["session_count"] == 1
    assert "very-secret-session" not in props["session_ids"]
    assert len(props["session_ids"]) == 16


@pytest.mark.asyncio
async def test_remote_client_passthrough_forwards_every_option(harness, monkeypatch):
    from unittest.mock import AsyncMock

    client = type("Client", (), {})()
    client.improve = AsyncMock(return_value={"legacy": "run"})
    state_mod = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(state_mod, "get_remote_client", lambda: client)

    result = await harness.improve_mod.improve(
        "docs",
        node_name=["Carlos"],
        session_ids=["s1"],
        build_global_context_index=True,
        build_truth_subspace=True,
        run_in_background=True,
    )

    client.improve.assert_awaited_once()
    args, kwargs = client.improve.await_args
    assert args == ("docs",)
    assert kwargs["node_name"] == ["Carlos"]
    assert kwargs["session_ids"] == ["s1"]
    assert kwargs["build_global_context_index"] is True
    assert kwargs["build_truth_subspace"] is True
    assert kwargs["run_in_background"] is True
    assert isinstance(result, ImproveResult)
    assert result.memify_run == {"legacy": "run"}
    assert harness.resolve_calls == []


@pytest.mark.asyncio
async def test_feedback_alpha_kwarg_overrides_config(harness):
    stage = FakeStage("a")
    harness.use_stages([stage])
    harness.set_config(feedback_alpha=0.3)

    await harness.improve()
    assert stage.seen_inputs[0].feedback_alpha == 0.3

    await harness.improve(feedback_alpha=0.7)
    assert stage.seen_inputs[1].feedback_alpha == 0.7
