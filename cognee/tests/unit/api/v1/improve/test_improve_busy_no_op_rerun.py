"""improve(session_ids=...) control flow after SDK-593.

Covers the four answers a session improve can give — busy, no_op, accepted and a
real run — and what each one costs: which stages run, whether an operation
record is opened, and how the per-session lock and rerun flag are handled. Every
stage, the probe, the lock and the recorder are stubbed; no cache, DB or LLM.
"""

import asyncio
import importlib
import types
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import ImproveLockRelease, ImproveLockStatus
from cognee.modules.session_bridge import SessionPendingWork

improve_mod = importlib.import_module("cognee.api.v1.improve.improve")
locks_pkg = importlib.import_module("cognee.infrastructure.locks")
gsm_mod = importlib.import_module("cognee.infrastructure.session.get_session_manager")
memify_pkg = importlib.import_module("cognee.modules.memify")
startup_mod = importlib.import_module("cognee.modules.migrations.startup")
state_mod = importlib.import_module("cognee.api.v1.serve.state")
utils_mod = importlib.import_module("cognee.shared.utils")


def _user():
    return types.SimpleNamespace(id=uuid4(), tenant_id=None)


class Harness:
    """Records what improve() touched."""

    def __init__(self):
        self.operations: list[str] = []
        self.operation_flags: list[dict] = []
        self.stages: list[str] = []
        self.memify_calls: list[dict] = []
        self.lock_calls: list[tuple] = []
        self.probe_results: list[dict[str, SessionPendingWork]] = []
        self.rerun_flags: list[bool] = []
        # lock state
        self.lock_free = True
        self.holder_age = 12.5


@pytest.fixture
def harness(monkeypatch):
    h = Harness()
    user = _user()
    dataset = types.SimpleNamespace(id=uuid4(), owner_id=user.id)

    async def fake_resolve(_dataset, _user):
        return _user, [dataset]

    monkeypatch.setattr(improve_mod, "resolve_authorized_user_datasets", fake_resolve)
    monkeypatch.setattr(state_mod, "get_remote_client", lambda: None)
    monkeypatch.setattr(utils_mod, "send_telemetry", lambda *a, **k: None)

    async def fake_migrations(*a, **k):
        return None

    monkeypatch.setattr(startup_mod, "run_migrations_and_block", fake_migrations)

    fake_sm = types.SimpleNamespace(is_available=True, is_auto_feedback_enabled=lambda: True)
    monkeypatch.setattr(gsm_mod, "get_session_manager", lambda: fake_sm)

    @asynccontextmanager
    async def fake_record_operation(name, **kwargs):
        h.operations.append(name)
        flags: dict = {}
        h.operation_flags.append(flags)
        yield types.SimpleNamespace(
            set_user=lambda u: flags.__setitem__("user", u),
            set_dataset=lambda d: flags.__setitem__("dataset", d),
            set_session_id=lambda s: flags.__setitem__("session_id", s),
            set_background=lambda b: flags.__setitem__("background", b),
        )

    monkeypatch.setattr(improve_mod, "record_operation", fake_record_operation)

    async def fake_probe(session_manager, *, user_id, session_ids):
        result = h.probe_results.pop(0)
        return {sid: result.get(sid, SessionPendingWork(sid)) for sid in session_ids}

    monkeypatch.setattr(improve_mod, "probe_sessions_pending_work", fake_probe)

    # Lock primitives (imported lazily from the package inside improve()).
    async def fake_acquire(session_id, user_id):
        h.lock_calls.append(("acquire", session_id))
        if not h.lock_free:
            return None
        h.lock_free = False
        return "tok"

    async def fake_release(session_id, user_id, token, *, force=False):
        # Mirrors the real three-step release: a pending rerun request is
        # consumed instead of letting go; force lets go regardless.
        if force:
            h.lock_calls.append(("release", session_id, token, "force"))
            h.lock_free = True
            return ImproveLockRelease.RELEASED
        if h.rerun_flags and h.rerun_flags[0]:
            h.rerun_flags.pop(0)
            h.lock_calls.append(("release_rerun", session_id))
            return ImproveLockRelease.RERUN
        h.lock_calls.append(("release", session_id, token))
        h.lock_free = True
        return ImproveLockRelease.RELEASED

    async def fake_request_rerun(session_id, user_id):
        h.lock_calls.append(("request_rerun", session_id))
        return ImproveLockStatus(
            busy=not h.lock_free, holder_age_seconds=h.holder_age, rerun_requested=True
        )

    monkeypatch.setattr(locks_pkg, "try_acquire_improve_lock", fake_acquire)
    monkeypatch.setattr(locks_pkg, "release_improve_lock", fake_release)
    monkeypatch.setattr(locks_pkg, "request_improve_rerun", fake_request_rerun)

    # Stages.
    async def fake_bridge(**kwargs):
        if kwargs.get("feedback_session_ids"):
            h.stages.append(f"feedback:{','.join(kwargs['feedback_session_ids'])}")
        if kwargs.get("persist_session_ids"):
            h.stages.append(f"persist:{','.join(kwargs['persist_session_ids'])}")

    async def fake_traces(**kwargs):
        h.stages.append(f"traces:{','.join(kwargs['session_ids'])}")

    async def fake_agent_context(*, session_ids, user):
        h.stages.append(f"agent_context:{','.join(session_ids)}")
        return dict.fromkeys(session_ids, 0)

    async def fake_distill(**kwargs):
        h.stages.append(f"distill:{','.join(kwargs['session_ids'])}")
        return 0

    async def fake_preferences(**kwargs):
        h.stages.append("preferences")

    monkeypatch.setattr(improve_mod, "_bridge_sessions", fake_bridge)
    monkeypatch.setattr(improve_mod, "_persist_session_traces", fake_traces)
    monkeypatch.setattr(improve_mod, "_extract_agent_context_per_session", fake_agent_context)
    monkeypatch.setattr(improve_mod, "_distill_sessions", fake_distill)
    monkeypatch.setattr(improve_mod, "_update_user_preferences", fake_preferences)

    async def fake_memify(**kwargs):
        h.memify_calls.append(kwargs)
        h.stages.append("memify")
        return {"run": "ok"}

    monkeypatch.setattr(memify_pkg, "memify", fake_memify)

    h.user = user
    h.dataset = dataset
    return h


def _pending(session_id, **flags):
    return {session_id: SessionPendingWork(session_id, **flags)}


@pytest.mark.asyncio
async def test_no_op_when_nothing_is_above_the_watermarks(harness):
    harness.probe_results.append(_pending("s1"))

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert result["status"] == "no_op"
    assert result["reason"] == "nothing_pending"
    assert result["session_ids"] == ["s1"]
    assert result["dataset_id"] == str(harness.dataset.id)
    # Zero cost: no operation row, no stage, no memify.
    assert harness.operations == []
    assert harness.stages == []
    assert harness.memify_calls == []
    # The lock was taken for the probe and released again.
    assert harness.lock_calls == [("acquire", "s1"), ("release", "s1", "tok")]


@pytest.mark.asyncio
async def test_busy_answer_requests_a_rerun_and_records_nothing(harness):
    harness.lock_free = False  # someone else holds it

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert result["status"] == "busy"
    assert result["session_id"] == "s1"
    assert result["holder_age_seconds"] == 12.5
    assert result["rerun_requested"] is True
    assert harness.operations == []
    assert harness.stages == []
    assert harness.lock_calls == [("acquire", "s1"), ("request_rerun", "s1")]
    assert harness.probe_results == []  # never probed


@pytest.mark.asyncio
async def test_lock_released_between_claim_and_rerun_request_is_taken(harness):
    """The holder finished between our two calls: claim it instead of reporting busy."""
    harness.probe_results.append(_pending("s1"))
    calls = {"n": 0}
    original_acquire = locks_pkg.try_acquire_improve_lock

    async def flaky_acquire(session_id, user_id):
        calls["n"] += 1
        if calls["n"] == 1:
            harness.lock_calls.append(("acquire", session_id))
            return None
        return await original_acquire(session_id, user_id)

    async def freed_request(session_id, user_id):
        harness.lock_calls.append(("request_rerun", session_id))
        return ImproveLockStatus(busy=False)

    locks_pkg.try_acquire_improve_lock = flaky_acquire
    locks_pkg.request_improve_rerun = freed_request

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert result["status"] == "no_op"
    assert harness.lock_calls[:3] == [
        ("acquire", "s1"),
        ("request_rerun", "s1"),
        ("acquire", "s1"),
    ]


@pytest.mark.asyncio
async def test_pending_work_runs_only_the_stages_that_have_it(harness):
    harness.probe_results.append(_pending("s1", new_qa=True, new_traces_to_persist=True))

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert result == {"run": "ok"}
    assert harness.operations == ["improve"]
    assert harness.operation_flags[0]["session_id"] == "s1"
    assert harness.operation_flags[0]["background"] is False
    assert harness.stages == ["persist:s1", "traces:s1", "preferences", "memify"]
    assert harness.lock_calls[-1] == ("release", "s1", "tok")


@pytest.mark.asyncio
async def test_multi_session_improve_narrows_each_stage_to_its_sessions(harness):
    harness.probe_results.append(
        {
            "a": SessionPendingWork("a", feedback_qas=True),
            "b": SessionPendingWork("b", distillable_entries=True),
            "c": SessionPendingWork("c"),
        }
    )

    await improve_mod.improve(dataset="ds", session_ids=["a", "b", "c"], user=harness.user)

    assert harness.stages == ["feedback:a", "distill:b", "preferences", "memify"]
    # Multi-session improves never take the single-session lock.
    assert harness.lock_calls == []
    assert "session_id" not in harness.operation_flags[0]


@pytest.mark.asyncio
async def test_opt_in_flags_force_a_full_run_even_when_nothing_is_pending(harness, monkeypatch):
    harness.probe_results.append(_pending("s1"))

    async def fake_gci(**kwargs):
        harness.stages.append("global_context_index")
        return True

    monkeypatch.setattr(improve_mod, "_build_global_context_index", fake_gci)

    result = await improve_mod.improve(
        dataset="ds", session_ids=["s1"], user=harness.user, build_global_context_index=True
    )

    assert result == {"run": "ok"}
    assert harness.stages == [
        "feedback:s1",
        "persist:s1",
        "traces:s1",
        "agent_context:s1",
        "distill:s1",
        "preferences",
        "memify",
        "global_context_index",
    ]


@pytest.mark.asyncio
async def test_rerun_request_makes_the_holder_run_one_more_pass_over_the_new_tail(harness):
    harness.probe_results.append(_pending("s1", new_qa=True))
    harness.rerun_flags = [True]  # one busy caller asked for a rerun
    harness.probe_results.append(_pending("s1", new_traces_to_persist=True))  # the newer tail

    await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert harness.stages == [
        "persist:s1",
        "preferences",
        "memify",
        "traces:s1",
        "preferences",
        "memify",
    ]
    assert [c[0] for c in harness.lock_calls] == [
        "acquire",
        "release_rerun",  # a request was pending: consumed, lock kept, one more pass
        "release",
    ]
    assert harness.operations == ["improve"]  # still ONE operation record


@pytest.mark.asyncio
async def test_rerun_with_nothing_new_pending_ends_the_loop(harness):
    harness.probe_results.append(_pending("s1", new_qa=True))
    harness.rerun_flags = [True]
    harness.probe_results.append(_pending("s1"))

    await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert harness.stages == ["persist:s1", "preferences", "memify"]


@pytest.mark.asyncio
async def test_rerun_passes_are_bounded(harness):
    harness.probe_results.extend([_pending("s1", new_qa=True)] * 10)
    harness.rerun_flags = [True] * 10

    await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert harness.stages.count("memify") == improve_mod.IMPROVE_MAX_RERUN_PASSES
    # Past the bound the lock is force-released; the flag stays for the next acquirer.
    assert harness.lock_calls[-1] == ("release", "s1", "tok", "force")
    assert harness.rerun_flags  # not consumed by us


@pytest.mark.asyncio
async def test_background_mode_returns_accepted_and_the_task_holds_then_releases_the_lock(
    harness,
):
    harness.probe_results.append(_pending("s1", new_qa=True, distillable_entries=True))
    started = asyncio.Event()
    release_gate = asyncio.Event()

    async def slow_distill(**kwargs):
        harness.stages.append("distill")
        started.set()
        await release_gate.wait()
        return 0

    improve_mod._distill_sessions = slow_distill

    result = await improve_mod.improve(
        dataset="ds", session_ids=["s1"], user=harness.user, run_in_background=True
    )

    assert result["status"] == "accepted"
    assert result["background"] is True
    assert result["pending_stages"] == ["distill_sessions", "persist_sessions"]
    # Returned while the stages are still running, lock still held.
    await asyncio.wait_for(started.wait(), timeout=1)
    assert ("release", "s1", "tok") not in harness.lock_calls
    assert harness.operations == ["improve"]
    assert harness.operation_flags[0]["background"] is True

    release_gate.set()
    await asyncio.gather(*improve_mod._BACKGROUND_IMPROVE_TASKS)

    assert harness.stages == ["persist:s1", "distill", "preferences", "memify"]
    assert harness.lock_calls[-1] == ("release", "s1", "tok")
    assert harness.memify_calls[0]["run_in_background"] is False  # ordered inside the task


@pytest.mark.asyncio
async def test_background_task_releases_the_lock_when_a_stage_raises(harness):
    harness.probe_results.append(_pending("s1", new_qa=True))

    async def exploding_memify(**kwargs):
        raise RuntimeError("boom")

    memify_pkg.memify = exploding_memify

    result = await improve_mod.improve(
        dataset="ds", session_ids=["s1"], user=harness.user, run_in_background=True
    )
    assert result["status"] == "accepted"
    await asyncio.gather(*improve_mod._BACKGROUND_IMPROVE_TASKS)

    assert harness.lock_calls[-1] == ("release", "s1", "tok", "force")


@pytest.mark.asyncio
async def test_blocking_stage_failure_still_releases_the_lock(harness):
    harness.probe_results.append(_pending("s1", new_qa=True))

    async def exploding_memify(**kwargs):
        raise RuntimeError("boom")

    memify_pkg.memify = exploding_memify

    with pytest.raises(RuntimeError):
        await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert harness.lock_calls[-1] == ("release", "s1", "tok", "force")


@pytest.mark.asyncio
async def test_session_cache_unavailable_is_reported_as_the_no_op_reason(harness, monkeypatch):
    fake_sm = types.SimpleNamespace(is_available=False, is_auto_feedback_enabled=lambda: True)
    monkeypatch.setattr(gsm_mod, "get_session_manager", lambda: fake_sm)

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert result["status"] == "no_op"
    assert result["reason"] == "session_cache_unavailable"
    assert harness.probe_results == []  # the probe is skipped entirely


@pytest.mark.asyncio
async def test_dataset_only_improve_is_unchanged(harness):
    result = await improve_mod.improve(dataset="ds", user=harness.user)

    assert result == {"run": "ok"}
    assert harness.operations == ["improve"]
    assert harness.stages == ["memify"]
    assert harness.lock_calls == []
    assert harness.probe_results == []


@pytest.mark.asyncio
async def test_precondition_failure_is_still_recorded_as_a_failed_improve(harness, monkeypatch):
    async def failing_resolve(_dataset, _user):
        raise ValueError("no such dataset")

    monkeypatch.setattr(improve_mod, "resolve_authorized_user_datasets", failing_resolve)

    with pytest.raises(ValueError):
        await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert harness.operations == ["improve"]
    assert harness.lock_calls == []


@pytest.mark.asyncio
async def test_rerun_request_landing_during_a_no_op_probe_turns_into_a_run(harness):
    """The race the conditional release closes: the probe found nothing, a busy caller
    then asked for a pass, and the release is refused — so the caller's tail is covered
    instead of being promised and dropped."""
    harness.probe_results.append(_pending("s1"))  # nothing pending at first
    harness.rerun_flags = [True]  # ...then a busy caller's request lands
    harness.probe_results.append(_pending("s1", new_qa=True))  # its newer tail

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    assert result == {"run": "ok"}  # a real run, not a no_op answer
    assert harness.stages == ["persist:s1", "preferences", "memify"]
    assert [c[0] for c in harness.lock_calls] == ["acquire", "release_rerun", "release"]
    assert harness.operations == ["improve"]


@pytest.mark.asyncio
async def test_rerun_landing_during_no_op_with_still_nothing_pending_ends_cleanly(harness):
    harness.probe_results.append(_pending("s1"))
    harness.rerun_flags = [True]
    harness.probe_results.append(_pending("s1"))  # the caller's data was in our snapshot

    result = await improve_mod.improve(dataset="ds", session_ids=["s1"], user=harness.user)

    # One operation record, no stages, lock released.
    assert result == {}
    assert harness.stages == []
    assert harness.operations == ["improve"]
    assert harness.lock_calls[-1] == ("release", "s1", "tok")
