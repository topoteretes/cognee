"""Orchestration tests for improve(): stage order, fail-open vs fatal stages, the
structured result contract, and the per-session improve lock.

Every pipeline is mocked at its source module (improve() imports them lazily), so
these tests exercise only the orchestration layer — no DB, cache, or LLM.
"""

import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


class DummySpan:
    def __init__(self):
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _wire(monkeypatch, *, feedback_influence=0.2, lock_acquired=True):
    """Patch every stage pipeline; return (order, mocks) for assertions."""
    improve_module = import_module("cognee.api.v1.improve.improve")
    shared_utils = import_module("cognee.shared.utils")
    serve_state = import_module("cognee.api.v1.serve.state")
    base_config = import_module("cognee.base_config")
    locks = import_module("cognee.infrastructure.locks")
    feedback_pipeline_mod = import_module("cognee.memify_pipelines.apply_feedback_weights")
    persist_pipeline_mod = import_module(
        "cognee.memify_pipelines.persist_sessions_in_knowledge_graph"
    )
    traces_pipeline_mod = import_module(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph"
    )
    gsm_mod = import_module("cognee.infrastructure.session.get_session_manager")
    ace_mod = import_module("cognee.infrastructure.session.agent_context_extraction")
    distillation_mod = import_module("cognee.modules.session_distillation")
    truth_build_mod = import_module("cognee.modules.truth_subspace.build")
    memify_mod = import_module("cognee.modules.memify")

    monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(improve_module, "new_span", lambda _: DummySpan())
    monkeypatch.setattr(
        base_config,
        "get_base_config",
        lambda: SimpleNamespace(default_feedback_influence=feedback_influence),
    )

    resolved = SimpleNamespace(id=uuid4(), name="docs")
    monkeypatch.setattr(
        improve_module,
        "resolve_authorized_user_datasets",
        AsyncMock(side_effect=lambda dataset, user: (user, [resolved])),
    )

    order = []

    def recorder(name, return_value=None, raises=None):
        async def _record(*args, **kwargs):
            order.append(name)
            if raises is not None:
                raise raises
            return return_value

        return AsyncMock(side_effect=_record)

    mocks = {
        "feedback_weights": recorder("feedback_weights"),
        "persist_sessions": recorder("persist_sessions"),
        "persist_traces": recorder("persist_traces"),
        "extract_agent_context": recorder("extract_agent_context", return_value=["lesson-id"]),
        "distill": recorder(
            "distill",
            return_value=SimpleNamespace(status="completed", documents=["doc"]),
        ),
        "truth_build": recorder(
            "truth_build",
            return_value={"anchors": 1, "nodes_scored": 3, "signature": "sig", "truth_epoch": 1},
        ),
        "memify": recorder("memify", return_value={"run": "info"}),
        "release_lock": AsyncMock(),
    }

    monkeypatch.setattr(
        feedback_pipeline_mod, "apply_feedback_weights_pipeline", mocks["feedback_weights"]
    )
    monkeypatch.setattr(
        persist_pipeline_mod,
        "persist_sessions_in_knowledge_graph_pipeline",
        mocks["persist_sessions"],
    )
    monkeypatch.setattr(
        traces_pipeline_mod,
        "persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        mocks["persist_traces"],
    )
    monkeypatch.setattr(
        gsm_mod,
        "get_session_manager",
        lambda: SimpleNamespace(is_available=True, is_auto_feedback_enabled=lambda: True),
    )
    monkeypatch.setattr(ace_mod, "extract_pending_agent_context", mocks["extract_agent_context"])
    monkeypatch.setattr(distillation_mod, "distill_session", mocks["distill"])
    monkeypatch.setattr(truth_build_mod, "build_truth_subspace", mocks["truth_build"])
    monkeypatch.setattr(memify_mod, "memify", mocks["memify"])
    monkeypatch.setattr(locks, "try_acquire_improve_lock", AsyncMock(return_value=lock_acquired))
    monkeypatch.setattr(locks, "release_improve_lock", mocks["release_lock"])

    return improve_module, order, mocks


def _user():
    return SimpleNamespace(id=uuid4())


@pytest.mark.asyncio
async def test_stage_order_and_structured_result(monkeypatch):
    improve_module, order, mocks = _wire(monkeypatch)

    result = await improve_module.improve(
        dataset="docs",
        session_ids=["s1"],
        build_truth_subspace=True,
        user=_user(),
    )

    assert order == [
        "feedback_weights",
        "persist_sessions",
        "persist_traces",
        "extract_agent_context",
        "distill",
        "truth_build",
        "memify",
    ]
    assert result["status"] == "completed"
    assert result["run_info"] == {"run": "info"}
    by_stage = {record["stage"]: record for record in result["stages"]}
    assert by_stage["feedback_weights"]["status"] == "ok"
    assert by_stage["persist_sessions"]["status"] == "ok"
    assert by_stage["persist_trace_steps"]["status"] == "ok"
    assert by_stage["extract_agent_context"] == {
        "stage": "extract_agent_context",
        "status": "ok",
        "error": None,
        "reason": None,
        "count": 1,
    }
    assert by_stage["distill_sessions"]["count"] == 1
    assert by_stage["build_truth_subspace"]["status"] == "ok"
    assert by_stage["build_truth_subspace"]["count"] == 3
    assert by_stage["memify_enrichment"]["status"] == "ok"
    mocks["release_lock"].assert_awaited_once_with("s1")


@pytest.mark.asyncio
async def test_feedback_alpha_is_plumbed_to_the_pipeline(monkeypatch):
    improve_module, _order, mocks = _wire(monkeypatch)

    await improve_module.improve(
        dataset="docs", session_ids=["s1"], user=_user(), feedback_alpha=0.5
    )

    assert mocks["feedback_weights"].await_args.kwargs["alpha"] == 0.5


@pytest.mark.asyncio
async def test_influence_zero_skips_feedback_weights_stage(monkeypatch):
    improve_module, order, mocks = _wire(monkeypatch, feedback_influence=0.0)

    result = await improve_module.improve(dataset="docs", session_ids=["s1"], user=_user())

    mocks["feedback_weights"].assert_not_awaited()
    assert "feedback_weights" not in order
    by_stage = {record["stage"]: record for record in result["stages"]}
    assert by_stage["feedback_weights"]["status"] == "skipped"
    assert by_stage["feedback_weights"]["reason"] == "influence_zero"


@pytest.mark.asyncio
async def test_persist_sessions_failure_is_fatal_and_releases_lock(monkeypatch):
    """Pins the current fail-closed behavior of the persist-Q&A stage.

    Unlike its fail-open siblings, a persist_sessions failure aborts every later
    stage (including memify enrichment). Whether that is the right contract is an
    open design question — if you change it, change this test consciously.
    """
    improve_module, _order, mocks = _wire(monkeypatch)
    mocks["persist_sessions"].side_effect = RuntimeError("persist blew up")

    with pytest.raises(RuntimeError, match="persist blew up"):
        await improve_module.improve(dataset="docs", session_ids=["s1"], user=_user())

    mocks["memify"].assert_not_awaited()
    mocks["release_lock"].assert_awaited_once_with("s1")


@pytest.mark.asyncio
async def test_lock_held_returns_explicit_skip(monkeypatch):
    improve_module, order, mocks = _wire(monkeypatch, lock_acquired=False)

    result = await improve_module.improve(dataset="docs", session_ids=["s1"], user=_user())

    assert result == {"status": "skipped", "reason": "lock_held", "stages": [], "run_info": None}
    assert order == []
    mocks["release_lock"].assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_trace_persistence_is_recorded_but_not_fatal(monkeypatch):
    improve_module, order, mocks = _wire(monkeypatch)
    mocks["persist_traces"].side_effect = RuntimeError("trace store down")

    result = await improve_module.improve(dataset="docs", session_ids=["s1"], user=_user())

    assert result["status"] == "completed"
    by_stage = {record["stage"]: record for record in result["stages"]}
    assert by_stage["persist_trace_steps"]["status"] == "failed"
    assert "trace store down" in by_stage["persist_trace_steps"]["error"]
    assert "memify" in order


@pytest.mark.asyncio
async def test_truth_build_skip_status_is_surfaced(monkeypatch):
    improve_module, _order, mocks = _wire(monkeypatch)
    mocks["truth_build"].side_effect = None
    mocks["truth_build"].return_value = {
        "anchors": 0,
        "nodes_scored": 0,
        "signature": "",
        "truth_epoch": 0,
        "skipped": "backend_unsupported",
    }

    result = await improve_module.improve(
        dataset="docs", session_ids=["s1"], build_truth_subspace=True, user=_user()
    )

    by_stage = {record["stage"]: record for record in result["stages"]}
    assert by_stage["build_truth_subspace"]["status"] == "skipped"
    assert by_stage["build_truth_subspace"]["reason"] == "backend_unsupported"


@pytest.mark.asyncio
async def test_concurrent_improves_serialize_on_the_real_lock(monkeypatch):
    """Exactly one of two concurrent single-session improves runs; the other skips."""
    improve_module, _order, mocks = _wire(monkeypatch)

    # Swap the mocked lock for the real in-process implementation.
    locks = import_module("cognee.infrastructure.locks")
    session_lock_mod = import_module("cognee.infrastructure.locks.session_lock")
    monkeypatch.setattr(
        locks, "try_acquire_improve_lock", session_lock_mod.try_acquire_improve_lock
    )
    monkeypatch.setattr(locks, "release_improve_lock", session_lock_mod.release_improve_lock)

    gate = asyncio.Event()

    async def blocking_memify(*args, **kwargs):
        await gate.wait()
        return {"run": "info"}

    mocks["memify"].side_effect = blocking_memify

    first = asyncio.create_task(
        improve_module.improve(dataset="docs", session_ids=["s-conc"], user=_user())
    )
    for _ in range(5):  # let the first improve claim the lock and block in memify
        await asyncio.sleep(0)

    second = await improve_module.improve(dataset="docs", session_ids=["s-conc"], user=_user())
    assert second == {"status": "skipped", "reason": "lock_held", "stages": [], "run_info": None}

    gate.set()
    first_result = await first
    assert first_result["status"] == "completed"

    # Lock is released: a follow-up improve acquires it again.
    third = await improve_module.improve(dataset="docs", session_ids=["s-conc"], user=_user())
    assert third["status"] == "completed"
