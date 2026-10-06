"""Typed QA/trace entries carry node_set through remember() (SDK-336).

The integrations plugins probe ``/openapi.json`` for a ``node_set`` property on
the ``QAEntry`` and ``TraceEntry`` schemas before tagging capture, so the
schema is part of the contract. The dispatcher pins the set before it writes
the entry and refuses a conflicting set without writing anything.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import cognee.api.v1.remember.remember  # registers the submodule; the package re-exports the function
from cognee.infrastructure.session.session_node_set import (
    SESSION_NODE_SET_STATE_ID,
    SESSION_NODE_SET_STATE_KIND,
    SessionNodeSetConflictError,
)
from cognee.memory.entries import QAEntry, TraceEntry

remember_module = sys.modules["cognee.api.v1.remember.remember"]


def test_openapi_advertises_node_set_on_qa_and_trace_entries():
    from fastapi import FastAPI

    from cognee.api.v1.remember.routers.get_remember_router import get_remember_router

    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/api/v1/remember")
    schemas = app.openapi()["components"]["schemas"]

    assert "node_set" in schemas["QAEntry"]["properties"]
    assert "node_set" in schemas["TraceEntry"]["properties"]


def _session_manager(pinned: list[str] | None = None) -> MagicMock:
    rows = (
        [{"id": SESSION_NODE_SET_STATE_ID, "kind": SESSION_NODE_SET_STATE_KIND, "node_set": pinned}]
        if pinned
        else []
    )
    manager = MagicMock()
    manager.is_available = True
    manager.get_session_context_entries = AsyncMock(return_value=rows)
    manager.create_session_context_entry = AsyncMock(return_value=True)
    manager.add_qa = AsyncMock(return_value="qa-1")
    manager.add_agent_trace_step = AsyncMock(return_value="trace-1")
    return manager


@pytest.fixture
def dispatch(monkeypatch):
    """Run ``_dispatch_session_entry`` against a fake session manager and no databases."""
    # The dispatcher imports these lazily by module path; the packages re-export
    # each function under its module's name, so patch the module objects.
    import cognee.infrastructure.session.get_session_manager
    import cognee.modules.data.methods.get_authorized_dataset
    import cognee.modules.engine.operations.setup
    import cognee.modules.session_lifecycle.metrics

    modules = {
        name: sys.modules[name]
        for name in (
            "cognee.infrastructure.session.get_session_manager",
            "cognee.modules.data.methods.get_authorized_dataset",
            "cognee.modules.engine.operations.setup",
            "cognee.modules.session_lifecycle.metrics",
        )
    }
    monkeypatch.setattr(modules["cognee.modules.engine.operations.setup"], "setup", AsyncMock())
    monkeypatch.setattr(
        modules["cognee.modules.data.methods.get_authorized_dataset"],
        "get_authorized_dataset",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        modules["cognee.modules.session_lifecycle.metrics"], "ensure_and_touch_session", AsyncMock()
    )

    async def run(entry, manager):
        monkeypatch.setattr(
            modules["cognee.infrastructure.session.get_session_manager"],
            "get_session_manager",
            lambda: manager,
        )
        return await remember_module._dispatch_session_entry(
            entry, dataset_name="d", session_id="s", user=SimpleNamespace(id=uuid4())
        )

    return run


@pytest.mark.asyncio
async def test_first_qa_entry_with_a_node_set_pins_the_session_then_stores(dispatch):
    manager = _session_manager()

    result = await dispatch(QAEntry(question="q", answer="a", node_set=["project-a"]), manager)

    assert result.status == "session_stored"
    pinned = manager.create_session_context_entry.await_args.kwargs["entry_dump"]
    assert pinned["node_set"] == ["project-a"]
    manager.add_qa.assert_awaited_once()


@pytest.mark.asyncio
async def test_trace_entry_with_the_pinned_set_stores_without_rewriting_the_pin(dispatch):
    manager = _session_manager(pinned=["project-a"])

    result = await dispatch(TraceEntry(origin_function="f", node_set=["project-a"]), manager)

    assert result.status == "session_stored"
    manager.create_session_context_entry.assert_not_awaited()
    manager.add_agent_trace_step.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_conflicting_node_set_is_refused_before_anything_is_written(dispatch):
    manager = _session_manager(pinned=["project-a"])

    with pytest.raises(SessionNodeSetConflictError):
        await dispatch(QAEntry(question="q", answer="a", node_set=["project-b"]), manager)

    manager.add_qa.assert_not_awaited()
    manager.create_session_context_entry.assert_not_awaited()


@pytest.mark.asyncio
async def test_entries_without_a_node_set_never_touch_the_pin(dispatch):
    manager = _session_manager()

    await dispatch(QAEntry(question="q", answer="a"), manager)

    manager.get_session_context_entries.assert_not_awaited()
    manager.create_session_context_entry.assert_not_awaited()


# ---------------------------------------------------------------- the field itself


def test_node_set_is_not_size_limited_like_the_call_level_node_set():
    """add()/update()/remember() never capped count or length; the entry must not either."""
    many = [f"project-{i}" for i in range(40)]
    long_name = "p" * 1000

    assert QAEntry(question="q", answer="a", node_set=many).node_set == sorted(many)
    assert TraceEntry(origin_function="f", node_set=[long_name]).node_set == [long_name]


def test_an_empty_node_set_name_is_refused():
    """An empty name would pin, and add() would create, a node set with no name."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QAEntry(question="q", answer="a", node_set=["project-a", ""])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("project-a", id="bare-string"),
        pytest.param([""], id="empty-name"),
        pytest.param(["   "], id="blank-name"),
        pytest.param(["project-a", 7], id="non-string-name"),
        pytest.param({"project-a"}, id="set-not-list"),
    ],
)
def test_the_field_refuses_what_the_shared_rule_refuses(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QAEntry(question="q", answer="a", node_set=bad)


def test_the_field_stores_the_set_sorted_and_deduplicated():
    entry = TraceEntry(origin_function="f", node_set=["b", "a", "b"])

    assert entry.node_set == ["a", "b"]
    assert TraceEntry(origin_function="f", node_set=[]).node_set == []
    assert TraceEntry(origin_function="f").node_set is None


def test_the_conflict_error_is_exported_from_the_package():
    import cognee

    assert cognee.SessionNodeSetConflictError is SessionNodeSetConflictError


# ------------------------------------------- the call-level kwarg on the session paths


def test_kwarg_fills_an_empty_entry_node_set():
    entry = QAEntry(question="q", answer="a")

    applied = remember_module._entry_with_node_set(entry, ["project-a"])

    assert applied.node_set == ["project-a"]
    assert entry.node_set is None, "the caller's entry is not mutated"


def test_kwarg_that_agrees_with_the_entry_changes_nothing():
    entry = TraceEntry(origin_function="f", node_set=["project-a", "project-b"])

    assert remember_module._entry_with_node_set(entry, ["project-b", "project-a"]) is entry


def test_kwarg_that_disagrees_with_the_entry_is_an_error():
    entry = QAEntry(question="q", answer="a", node_set=["project-a"])

    with pytest.raises(ValueError, match="node_set given twice"):
        remember_module._entry_with_node_set(entry, ["project-b"])


def test_kwarg_on_an_entry_that_cannot_carry_a_node_set_is_an_error():
    from cognee.memory.entries import FeedbackEntry

    with pytest.raises(TypeError, match="not supported for FeedbackEntry"):
        remember_module._entry_with_node_set(
            FeedbackEntry(qa_id="qa-1", feedback_score=5), ["project-a"]
        )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("project-a", id="bare-string"),
        pytest.param([""], id="empty-name"),
        pytest.param(["  "], id="blank-name"),
    ],
)
def test_kwarg_is_validated_by_the_same_rule_as_the_field(bad):
    """The kwarg used to skip validation: [""] was stored and a string was split."""
    with pytest.raises(ValueError, match="node_set"):
        remember_module._entry_with_node_set(QAEntry(question="q", answer="a"), bad)


def test_kwarg_fill_is_normalized_like_the_field():
    applied = remember_module._entry_with_node_set(
        QAEntry(question="q", answer="a"), ["b", "a", "b"]
    )

    assert applied.node_set == ["a", "b"]


def test_an_empty_kwarg_means_no_node_set_and_changes_nothing():
    from cognee.memory.entries import FeedbackEntry

    entry = QAEntry(question="q", answer="a", node_set=["project-a"])
    feedback = FeedbackEntry(qa_id="qa-1", feedback_score=5)

    assert remember_module._entry_with_node_set(entry, []) is entry
    assert remember_module._entry_with_node_set(feedback, []) is feedback


@pytest.mark.asyncio
async def test_remember_refuses_an_invalid_kwarg_before_dispatching(monkeypatch):
    dispatched = AsyncMock()
    monkeypatch.setattr(remember_module, "_remember_entry", dispatched)

    with pytest.raises(ValueError, match="not a single string"):
        await remember_module.remember(
            QAEntry(question="q", answer="a"), session_id="s", node_set="project-a"
        )

    dispatched.assert_not_awaited()


@pytest.mark.asyncio
async def test_remember_applies_the_kwarg_before_dispatching_a_typed_entry(monkeypatch):
    """``remember(entry, session_id=..., node_set=[...])`` must not drop the kwarg."""
    seen = {}

    async def fake_remember_entry(entry, **kwargs):
        seen["entry"] = entry
        return SimpleNamespace(status="session_stored")

    monkeypatch.setattr(remember_module, "_remember_entry", fake_remember_entry)

    await remember_module.remember(
        QAEntry(question="q", answer="a"), session_id="s", node_set=["project-a"]
    )

    assert seen["entry"].node_set == ["project-a"]


@pytest.mark.asyncio
async def test_plain_text_with_a_node_set_pins_the_session_before_the_write(monkeypatch):
    gsm = sys.modules["cognee.infrastructure.session.get_session_manager"]

    manager = _session_manager()
    monkeypatch.setattr(gsm, "get_session_manager", lambda: manager)
    calls = []
    manager.create_session_context_entry.side_effect = lambda **kw: calls.append("pin") or True
    manager.add_qa.side_effect = lambda **kw: calls.append("add_qa") or "qa-1"

    await remember_module._add_to_session(
        "s", "a fact", SimpleNamespace(id=uuid4()), node_set=["project-a"]
    )

    assert calls == ["pin", "add_qa"]
    assert manager.create_session_context_entry.await_args.kwargs["entry_dump"]["node_set"] == [
        "project-a"
    ]
    assert manager.add_qa.await_args.kwargs["answer"] == "a fact"


@pytest.mark.asyncio
async def test_plain_text_with_a_conflicting_node_set_writes_nothing(monkeypatch):
    gsm = sys.modules["cognee.infrastructure.session.get_session_manager"]

    manager = _session_manager(pinned=["project-a"])
    monkeypatch.setattr(gsm, "get_session_manager", lambda: manager)

    with pytest.raises(SessionNodeSetConflictError):
        await remember_module._add_to_session(
            "s", "a fact", SimpleNamespace(id=uuid4()), node_set=["project-b"]
        )

    manager.add_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_text_without_a_node_set_never_touches_the_pin(monkeypatch):
    gsm = sys.modules["cognee.infrastructure.session.get_session_manager"]

    manager = _session_manager()
    monkeypatch.setattr(gsm, "get_session_manager", lambda: manager)

    await remember_module._add_to_session("s", "a fact", SimpleNamespace(id=uuid4()))

    manager.get_session_context_entries.assert_not_awaited()
    manager.add_qa.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [pytest.param("project-a", id="bare-string"), pytest.param([""], id="empty-name")],
)
async def test_plain_text_refuses_an_invalid_node_set_and_writes_nothing(monkeypatch, bad):
    """A bare string used to pin one node set per letter of the name."""
    gsm = sys.modules["cognee.infrastructure.session.get_session_manager"]
    manager = _session_manager()
    monkeypatch.setattr(gsm, "get_session_manager", lambda: manager)

    with pytest.raises(ValueError, match="node_set"):
        await remember_module._add_to_session(
            "s", "a fact", SimpleNamespace(id=uuid4()), node_set=bad
        )

    manager.create_session_context_entry.assert_not_awaited()
    manager.add_qa.assert_not_awaited()
