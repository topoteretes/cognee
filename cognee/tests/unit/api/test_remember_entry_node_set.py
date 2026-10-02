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
