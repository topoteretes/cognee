"""``RememberResult.improve`` / ``.improve_error`` (plan item A5) and the
``IMPROVE_AUTO_ENABLED`` kill switch, plus hashed session ids in telemetry (A6).

A failed automatic ``improve()`` after a successful cognify is reported on the
result and never flips the remember's status to ``errored``.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.improve.config import ImproveConfig
from cognee.modules.improve.result import ImproveResult, StageResult

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
debounce_module = importlib.import_module("cognee.api.v1.remember.auto_improve_debounce")
improve_pkg = importlib.import_module("cognee.api.v1.improve")


@pytest.fixture(autouse=True)
def _no_db_setup(monkeypatch):
    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)
    # Defaults: auto-improve on, no debounce.
    monkeypatch.setattr(debounce_module, "get_improve_config", lambda: ImproveConfig())


@pytest.fixture
def permanent_pipeline(monkeypatch):
    """Stub add()/cognify() so the permanent path runs without databases."""
    calls = {"add": 0, "cognify": 0}

    async def fake_add(*args, **kwargs):
        calls["add"] += 1

    async def fake_cognify(*args, **kwargs):
        calls["cognify"] += 1
        return {}

    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)
    return calls


def _completed_improve() -> ImproveResult:
    return ImproveResult(
        stages=[StageResult.completed("triplet_enrichment", edges=3)],
        memify_run={},
    )


@pytest.mark.asyncio
async def test_permanent_remember_attaches_improve_result(monkeypatch, permanent_pipeline):
    improve_result = _completed_improve()
    seen = {}

    async def fake_improve(**kwargs):
        seen.update(kwargs)
        return improve_result

    monkeypatch.setattr(improve_pkg, "improve", fake_improve)

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), user=SimpleNamespace(id=uuid4())
    )

    assert result.status == "completed"
    assert result.improve is improve_result
    assert result.improve_error is None
    assert result.to_dict()["improve"]["status"] == "completed"
    assert "improve_error" not in result.to_dict()
    assert "session_ids" not in seen


@pytest.mark.asyncio
async def test_improve_failure_keeps_remember_completed(monkeypatch, permanent_pipeline):
    async def failing_improve(**kwargs):
        raise RuntimeError("enrichment exploded")

    monkeypatch.setattr(improve_pkg, "improve", failing_improve)

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), user=SimpleNamespace(id=uuid4())
    )

    assert result.status == "completed"
    assert result.error is None
    assert bool(result) is True
    assert result.improve is None
    assert result.improve_error == "enrichment exploded"
    assert result.to_dict()["improve_error"] == "enrichment exploded"
    assert "improve_error='enrichment exploded'" in repr(result)


@pytest.mark.asyncio
async def test_errored_stage_is_reported_not_promoted(monkeypatch, permanent_pipeline):
    improve_result = ImproveResult(
        stages=[
            StageResult.completed("feedback_weights"),
            StageResult.errored("triplet_enrichment", "embedding backend down"),
        ],
        memify_run={},
    )

    async def fake_improve(**kwargs):
        return improve_result

    monkeypatch.setattr(improve_pkg, "improve", fake_improve)

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), user=SimpleNamespace(id=uuid4())
    )

    assert result.status == "completed"
    assert result.improve is improve_result
    assert result.improve.status == "errored"
    assert result.improve_error == "triplet_enrichment: embedding backend down"


@pytest.mark.asyncio
async def test_background_remember_records_improve_error(monkeypatch, permanent_pipeline):
    async def failing_improve(**kwargs):
        raise RuntimeError("late failure")

    monkeypatch.setattr(improve_pkg, "improve", failing_improve)

    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        run_in_background=True,
        user=SimpleNamespace(id=uuid4()),
    )
    # The stubs never block, so the task may already have finished by the time
    # remember() returns; awaiting is what the caller does either way.
    await result

    assert result.status == "completed"
    assert result.improve_error == "late failure"


@pytest.mark.asyncio
async def test_session_bridge_records_improve_error(monkeypatch):
    async def fake_add_to_session(session_id, data, user):
        return None

    async def failing_improve(**kwargs):
        raise RuntimeError("bridge failed")

    monkeypatch.setattr(remember_module, "_add_to_session", fake_add_to_session)
    monkeypatch.setattr(improve_pkg, "improve", failing_improve)

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), session_id="s-a5", user=SimpleNamespace(id=uuid4())
    )
    assert result.status == "session_stored"
    await result

    assert result.status == "session_stored"
    assert result.error is None
    assert result.improve is None
    assert result.improve_error == "bridge failed"


@pytest.mark.asyncio
async def test_session_bridge_attaches_improve_result(monkeypatch):
    improve_result = _completed_improve()

    async def fake_add_to_session(session_id, data, user):
        return None

    async def fake_improve(**kwargs):
        assert kwargs["session_ids"] == ["s-ok"]
        return improve_result

    monkeypatch.setattr(remember_module, "_add_to_session", fake_add_to_session)
    monkeypatch.setattr(improve_pkg, "improve", fake_improve)

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), session_id="s-ok", user=SimpleNamespace(id=uuid4())
    )
    await result

    assert result.improve is improve_result
    assert result.improve_error is None


@pytest.mark.asyncio
async def test_auto_enabled_false_disables_both_paths(monkeypatch, permanent_pipeline):
    monkeypatch.setattr(
        debounce_module, "get_improve_config", lambda: ImproveConfig(auto_enabled=False)
    )
    calls = {"improve": 0}

    async def counting_improve(**kwargs):
        calls["improve"] += 1
        return _completed_improve()

    async def fake_add_to_session(session_id, data, user):
        return None

    monkeypatch.setattr(improve_pkg, "improve", counting_improve)
    monkeypatch.setattr(remember_module, "_add_to_session", fake_add_to_session)
    user = SimpleNamespace(id=uuid4())

    permanent = await remember_module.remember("note", dataset_id=uuid4(), user=user)
    assert permanent.status == "completed"
    assert permanent.improve is None
    assert permanent.improve_error is None

    session = await remember_module.remember(
        "note", dataset_id=uuid4(), session_id="s-off", user=user
    )
    assert session.status == "session_stored"
    assert session._task is None
    await session

    assert calls["improve"] == 0
    assert permanent_pipeline["cognify"] == 1


@pytest.mark.asyncio
async def test_telemetry_never_carries_raw_session_ids(monkeypatch, permanent_pipeline):
    events = []

    def fake_send_telemetry(event_name, user, additional_properties=None, **kwargs):
        events.append((event_name, dict(additional_properties or {})))

    async def fake_add_to_session(session_id, data, user):
        return None

    async def fake_improve(**kwargs):
        return _completed_improve()

    monkeypatch.setattr("cognee.shared.utils.send_telemetry", fake_send_telemetry)
    monkeypatch.setattr(remember_module, "_add_to_session", fake_add_to_session)
    monkeypatch.setattr(improve_pkg, "improve", fake_improve)
    user = SimpleNamespace(id=uuid4())

    session_result = await remember_module.remember(
        "note", dataset_id=uuid4(), session_id="alice-private-chat", user=user
    )
    await session_result
    await remember_module.remember(
        "note", dataset_id=uuid4(), user=user, session_ids=["alice-private-chat", "bob-chat"]
    )

    remember_events = [props for name, props in events if name == "cognee.remember"]
    assert len(remember_events) == 2
    session_props, permanent_props = remember_events
    expected = remember_module._hash_session_id("alice-private-chat")
    assert session_props["session_id"] == expected
    assert len(expected) == 16
    assert permanent_props["session_id"] == ""
    assert permanent_props["session_ids"] == ",".join(
        remember_module._hash_session_id(sid) for sid in ["alice-private-chat", "bob-chat"]
    )
    for props in remember_events:
        blob = " ".join(str(value) for value in props.values())
        assert "alice-private-chat" not in blob
        assert "bob-chat" not in blob


def test_remote_payload_rebuilds_improve_result():
    result = remember_module.RememberResult(status="completed", dataset_name="d")
    payload = {
        "improve": _completed_improve().model_dump(mode="json"),
        "improve_error": "triplet_enrichment: boom",
    }
    result._attach_improve_payload(payload)

    assert isinstance(result.improve, ImproveResult)
    assert result.improve.stage("triplet_enrichment").status == "completed"
    assert result.improve_error == "triplet_enrichment: boom"
