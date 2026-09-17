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


def _lock_held(result: ImproveResult) -> bool:
    return bool(result.stages) and all(
        stage.status == "skipped" and stage.reason == REASON_LOCK_HELD for stage in result.stages
    )


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
            FakeStage("needs_sessions", needs_sessions=True, calls=calls),
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
async def test_unknown_disabled_stage_name_fails_loudly(harness):
    harness.use_stages([FakeStage("real_stage")])
    harness.set_config(stages_disabled=["real_stag"])  # one typo

    with pytest.raises(ValueError, match="real_stag"):
        await harness.improve()


@pytest.mark.asyncio
async def test_fatal_stage_cannot_be_disabled_by_config(harness):
    calls = []
    harness.use_stages([FakeStage("fatal_one", fatal=True, calls=calls)])
    harness.set_config(stages_disabled=["fatal_one"])

    with pytest.raises(ValueError, match="fatal"):
        await harness.improve()

    assert calls == []


@pytest.mark.asyncio
async def test_session_stage_runs_when_session_ids_given(harness):
    calls = []
    harness.use_stages([FakeStage("needs_sessions", needs_sessions=True, calls=calls)])

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
    assert _lock_held(result)
    assert result.status == "skipped"
    assert [(s.stage, s.status, s.reason) for s in result.stages] == [
        ("a", "skipped", REASON_LOCK_HELD),
        ("b", "skipped", REASON_LOCK_HELD),
    ]
    assert calls == []
    assert result.memify_run == {}
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == "a=skipped,b=skipped"


@pytest.mark.asyncio
async def test_lock_is_keyed_to_session_ids_and_the_dataset(harness):
    calls = []
    harness.use_stages([FakeStage("a", calls=calls)])
    # Another run holding one of our sessions blocks us. Session keys carry
    # the user id: session state is scoped per (user, session) everywhere.
    session_key = f"session:{harness.user.id}:chat_2"
    assert await session_lock.try_acquire_improve_lock_many([session_key])
    try:
        blocked = await harness.improve(session_ids=["chat_1", "chat_2"])
    finally:
        await session_lock.release_improve_lock_many([session_key])
    assert _lock_held(blocked)

    # A DIFFERENT user's session of the same name never blocks us.
    other_users_key = f"session:{uuid4()}:chat_1"
    assert await session_lock.try_acquire_improve_lock_many([other_users_key])
    try:
        allowed = await harness.improve(session_ids=["chat_1"])
    finally:
        await session_lock.release_improve_lock_many([other_users_key])
    assert not _lock_held(allowed)

    # So does a dataset-keyed run over the same dataset: a session-keyed
    # bridge and a plain improve(dataset=...) must never write concurrently.
    dataset_key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([dataset_key])
    try:
        blocked = await harness.improve(session_ids=["chat_1", "chat_2"])
    finally:
        await session_lock.release_improve_lock_many([dataset_key])
    assert _lock_held(blocked)

    allowed = await harness.improve(session_ids=["chat_1", "chat_2"])
    assert not _lock_held(allowed)
    assert calls == ["a", "a"]  # the other-user run above plus this one
    # And every claim — sessions and dataset — is released afterwards.
    all_keys = [
        f"session:{harness.user.id}:chat_1",
        f"session:{harness.user.id}:chat_2",
        dataset_key,
    ]
    assert await session_lock.try_acquire_improve_lock_many(all_keys)
    await session_lock.release_improve_lock_many(all_keys)


@pytest.mark.asyncio
async def test_fatal_stage_stops_run_and_raises_with_partial_result(harness):
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
async def test_fatal_stage_reporting_errored_run_info_also_stops_run(harness):
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
    assert result.stages[0].duration_ms >= 0


@pytest.mark.asyncio
async def test_background_mode_runs_all_stages_under_one_lock(harness):
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
    # The whole run, not one stage, is what holds the claim.
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
async def test_clean_foreground_run_leaves_the_operation_close_to_the_recorder(harness):
    harness.use_stages([FakeStage("a")])

    await harness.improve()

    operation = harness.operations[-1]
    assert operation.close_deferred is False
    assert operation.outcome is None  # record_operation writes "succeeded" on clean exit
    assert harness.finish_calls == []


@pytest.mark.asyncio
async def test_errored_non_fatal_run_marks_the_operation_failed(harness):
    from cognee.modules.pipelines.models import OperationOutcome

    harness.use_stages(
        [
            FakeStage("flaky", run=lambda _i: ValueError("nope")),
            FakeStage("next"),
        ]
    )

    result = await harness.improve()

    assert result.status == "errored"
    # The stage-8 watermark trusts the operation row, so a run with an errored
    # stage must not be recorded as a succeeded improve.
    assert harness.operations[-1].outcome == OperationOutcome.FAILED


@pytest.mark.asyncio
async def test_lock_held_run_records_a_noop_operation_row(harness):
    """A lost claim runs zero stages, so its row must never serve as a watermark.

    The dataset is bound to the record before the claim, so a "succeeded" row
    here would mark a dataset current that this call never improved — including
    one whose claim was lost on a shared *session* key.
    """
    from cognee.modules.pipelines.models import OperationOutcome

    harness.use_stages([FakeStage("a")])
    key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([key])
    try:
        result = await harness.improve()
    finally:
        await session_lock.release_improve_lock_many([key])

    assert _lock_held(result)
    assert harness.operations[-1].outcome == OperationOutcome.NOOP


@pytest.mark.asyncio
async def test_all_skipped_run_records_a_noop_operation_row(harness):
    """Nothing ran (e.g. sessionless with triplet_embedding off): not a watermark.

    Otherwise enabling triplet_embedding later finds "no writes since the last
    succeeded improve" and never enriches the data ingested before the flip.
    """
    from cognee.modules.pipelines.models import OperationOutcome

    harness.use_stages(
        [FakeStage("a", gate_reason="nope"), FakeStage("b", gate_reason="nope")],
    )

    result = await harness.improve()

    assert result.status == "skipped"
    assert harness.operations[-1].outcome == OperationOutcome.NOOP


@pytest.mark.asyncio
async def test_background_run_defers_the_operation_close_until_the_run_ends(harness):
    release = asyncio.Event()

    async def slow_stage(_inputs):
        await release.wait()
        return StageResult.completed("slow")

    harness.use_stages([FakeStage("slow", run=slow_stage)])

    result = await harness.improve(run_in_background=True)

    operation = harness.operations[-1]
    assert operation.close_deferred is True
    await asyncio.sleep(0)
    assert harness.finish_calls == []  # no row at launch: the run has not ended

    release.set()
    await result.wait()

    assert [call["context"] for call in harness.finish_calls] == [operation]
    assert harness.finish_calls[0]["error"] is None
    # The stage the run's own operation id was handed to (via inputs) matches.
    stage = harness.improve_mod.DEFAULT_STAGES[0]
    assert stage.seen_inputs[0].improve_operation_id == operation.operation_id


@pytest.mark.asyncio
async def test_background_fatal_closes_the_operation_with_the_error(harness):
    harness.use_stages(
        [FakeStage("fatal_one", fatal=True, run=lambda _i: RuntimeError("boom"))],
    )

    result = await harness.improve(run_in_background=True)
    await result.wait()

    assert len(harness.finish_calls) == 1
    assert isinstance(harness.finish_calls[0]["error"], RuntimeError)


@pytest.mark.asyncio
async def test_cancelled_background_run_still_closes_the_operation(harness):
    """CancelledError is not an Exception: it must propagate, yet the deferred
    row write still has to happen or the run leaves no trace in pipeline_runs."""
    running = asyncio.Event()

    async def blocked_stage(_inputs):
        running.set()
        await asyncio.Event().wait()  # blocks until cancelled

    harness.use_stages([FakeStage("slow", run=blocked_stage)])

    result = await harness.improve(run_in_background=True)
    await running.wait()
    result._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await result._task

    assert [call["context"] for call in harness.finish_calls] == [harness.operations[-1]]
    assert isinstance(harness.finish_calls[0]["error"], asyncio.CancelledError)
    # execute_stages' own finally still released the lock during cancellation.
    key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([key])
    await session_lock.release_improve_lock_many([key])


@pytest.mark.asyncio
async def test_telemetry_puts_session_ids_under_the_sanitized_key(harness):
    """send_telemetry hashes session ids centrally (TELEMETRY_SANITIZED_PROPERTIES);
    the call site's job is to put them ONLY under a key that rule covers."""
    from cognee.shared.utils import TELEMETRY_SANITIZED_PROPERTIES, _sanitize_nested_properties

    harness.use_stages([FakeStage("a")])

    await harness.improve(session_ids=["very-secret-session"])

    props = harness.telemetry[0]["properties"]
    assert props["session_count"] == 1
    assert props["session_ids"] == "very-secret-session"
    assert "session_ids" in TELEMETRY_SANITIZED_PROPERTIES
    sanitized = _sanitize_nested_properties(props, TELEMETRY_SANITIZED_PROPERTIES)
    assert "very-secret-session" not in " ".join(str(value) for value in sanitized.values())


@pytest.mark.asyncio
async def test_remote_client_passthrough_forwards_every_option(harness, monkeypatch):
    from unittest.mock import AsyncMock

    client = type("Client", (), {})()
    client.improve = AsyncMock(return_value={"legacy": "run"})
    monkeypatch.setattr(harness.improve_mod, "get_remote_client", lambda: client)

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
async def test_remote_result_stamps_the_span(harness, monkeypatch):
    """The remote path exits through report() like every local path, so the
    span carries the server's stage summary instead of staying unset."""
    from unittest.mock import AsyncMock

    payload = ImproveResult(
        stages=[StageResult.completed("a"), StageResult.skipped("b", "opt_in_disabled")],
        memify_run={},
    ).model_dump(mode="json")
    client = type("Client", (), {})()
    client.improve = AsyncMock(return_value=payload)
    monkeypatch.setattr(harness.improve_mod, "get_remote_client", lambda: client)

    result = await harness.improve_mod.improve("docs")

    assert result.stage("a").status == "completed"
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == "a=completed,b=skipped"


@pytest.mark.asyncio
async def test_feedback_alpha_kwarg_overrides_config(harness):
    stage = FakeStage("a")
    harness.use_stages([stage])
    harness.set_config(feedback_alpha=0.3)

    await harness.improve()
    assert stage.seen_inputs[0].feedback_alpha == 0.3

    await harness.improve(feedback_alpha=0.7)
    assert stage.seen_inputs[1].feedback_alpha == 0.7


def test_memify_passthrough_keys_are_declared_on_improve_kwargs():
    """The forwarded memify surface is spelled twice — MEMIFY_PASSTHROUGH_KEYS
    (what _resolve_inputs forwards) and ImproveKwargs (what type checkers let a
    caller pass). A key added to one and not the other is silently either
    rejected by type checkers or never forwarded; this pins the two together."""
    from cognee.api.v1.improve.improve import ImproveKwargs
    from cognee.modules.improve import MEMIFY_PASSTHROUGH_KEYS

    assert set(MEMIFY_PASSTHROUGH_KEYS) <= set(ImproveKwargs.__annotations__)


@pytest.mark.asyncio
async def test_lock_loser_never_probes_the_graph_engine(harness):
    """The capability probe leases the graph engine (and a dataset-queue slot);
    a run that loses its lock claim runs nothing, so it must not pay for it."""
    harness.use_stages([FakeStage("a")])
    resolve_mock = harness.improve_mod.resolve_graph_capabilities  # AsyncMock in conftest
    resolve_mock.reset_mock()
    key = f"dataset:{harness.dataset.id}"
    assert await session_lock.try_acquire_improve_lock_many([key])
    try:
        blocked = await harness.improve()
    finally:
        await session_lock.release_improve_lock_many([key])

    assert _lock_held(blocked)
    resolve_mock.assert_not_awaited()

    winner = await harness.improve()
    assert not _lock_held(winner)
    resolve_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_during_the_probe_releases_the_lock(harness):
    """The probe runs after the claim but before execute_stages' finally
    exists; a task cancelled there (client disconnect, wait_for timeout) must
    not strand the keys, or every later improve on the dataset skips
    lock_held until restart."""
    harness.use_stages([FakeStage("a")])
    probe_mock = harness.improve_mod.resolve_graph_capabilities  # AsyncMock in conftest
    probing = asyncio.Event()

    async def blocked_probe(_dataset_id, _owner_id):
        probing.set()
        await asyncio.Event().wait()  # blocks until cancelled

    harness.monkeypatch.setattr(harness.improve_mod, "resolve_graph_capabilities", blocked_probe)

    task = asyncio.create_task(harness.improve())
    await probing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    harness.monkeypatch.setattr(harness.improve_mod, "resolve_graph_capabilities", probe_mock)
    retry = await harness.improve()
    assert not _lock_held(retry)


def _stamped_stage(name: str, stamp: dict) -> FakeStage:
    def run(_inputs):
        result = StageResult.completed(name)
        result._run_info_stamp = stamp
        return result

    return FakeStage(name, run=run)


@pytest.mark.asyncio
async def test_stage_stamps_are_merged_onto_the_operation_row(harness):
    """The generic channel behind the stage-8 watermark: which runs stamp is
    the stage's decision (tested with the stage); the loop only merges what a
    stage asked for onto the row the run owns."""
    harness.use_stages(
        [
            _stamped_stage("s1", {"s1": {"status": "completed"}}),
            FakeStage("s2"),
            _stamped_stage("s3", {"s3": {"status": "already_completed"}}),
        ]
    )

    await harness.improve()

    assert harness.operations[-1].run_info == {
        "s1": {"status": "completed"},
        "s3": {"status": "already_completed"},
    }


@pytest.mark.asyncio
async def test_stamp_merge_is_append_style(harness):
    """A later stamp never drops an earlier stage's entry; reusing a key is
    last-writer-wins (and logged by merge_run_info as a namespace clash)."""
    harness.use_stages(
        [
            _stamped_stage("s1", {"shared": {"by": "s1"}, "s1": {"ok": True}}),
            _stamped_stage("s2", {"shared": {"by": "s2"}}),
        ]
    )

    await harness.improve()

    assert harness.operations[-1].run_info == {
        "s1": {"ok": True},
        "shared": {"by": "s2"},
    }


@pytest.mark.asyncio
async def test_stages_without_a_stamp_leave_the_row_unstamped(harness):
    """No stage stamped, so the row must carry no run_info: a leftover stamp
    would stand in as the enrichment watermark for work that never ran."""
    harness.use_stages([FakeStage("s1"), FakeStage("s2")])

    await harness.improve()

    assert not harness.operations[-1].run_info


@pytest.mark.asyncio
async def test_background_run_carries_the_stamp_on_the_deferred_row(harness):
    """The deferred row is written by _run_detached from the same context the
    stages stamped; losing the stamp there would unmoor the watermark."""
    harness.use_stages([_stamped_stage("s1", {"s1": {"status": "completed"}})])

    result = await harness.improve(run_in_background=True)
    await result.wait()

    assert harness.operations[-1].run_info == {"s1": {"status": "completed"}}
    assert harness.finish_calls[-1]["context"] is harness.operations[-1]
