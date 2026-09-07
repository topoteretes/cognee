"""The retrievers ask SessionManager.is_auto_feedback_enabled() instead of re-reading config."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.agentic_retriever import AgenticRetriever
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever

# The ``session`` package re-exports ``get_session_manager`` under the same name as
# its submodule, so both a dotted ``patch("...get_session_manager.get_session_manager")``
# (Python 3.10 resolves it by getattr) and ``import ... as module`` land on the
# re-exported function. ``import_module`` returns the module object itself.
get_session_manager_module = importlib.import_module(
    "cognee.infrastructure.session.get_session_manager"
)


def _manager(*, available=True, auto_feedback=True):
    return SimpleNamespace(
        is_available=available,
        is_auto_feedback_enabled=lambda: auto_feedback,
        resolve_session_id=lambda session_id: session_id or "default_session",
    )


@pytest.mark.asyncio
async def test_cot_block_skips_when_manager_reports_auto_feedback_off():
    retriever = SimpleNamespace(session_id="s1")
    with patch(
        "cognee.infrastructure.session.session_context_builder.build_active_context_block",
        new=AsyncMock(return_value=("BLOCK", ["e1"])),
    ) as build:
        block = await GraphCompletionCotRetriever._maybe_active_context_block(
            retriever, _manager(auto_feedback=False), "u1", "q"
        )
    assert block == ""
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_cot_block_renders_when_manager_reports_auto_feedback_on():
    retriever = SimpleNamespace(session_id="s1")
    with patch(
        "cognee.infrastructure.session.session_context_builder.build_active_context_block",
        new=AsyncMock(return_value=("BLOCK", ["e1"])),
    ) as build:
        block = await GraphCompletionCotRetriever._maybe_active_context_block(
            retriever, _manager(auto_feedback=True), "u1", "q"
        )
    assert block == "BLOCK"
    build.assert_awaited_once()


def _agentic(session_id="s1"):
    return SimpleNamespace(
        user=SimpleNamespace(id="u1"),
        session_id=session_id,
        _use_session_cache=lambda: True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("available", "auto_feedback"),
    [(True, False), (False, True)],
)
async def test_agentic_block_skips_when_manager_gate_is_closed(available, auto_feedback):
    manager = _manager(available=available, auto_feedback=auto_feedback)
    with (
        patch.object(
            get_session_manager_module,
            "get_session_manager",
            return_value=manager,
        ),
        patch(
            "cognee.infrastructure.session.session_context_builder.build_active_context_block",
            new=AsyncMock(return_value=("BLOCK", ["e1"])),
        ) as build,
    ):
        result = await AgenticRetriever._maybe_active_context_block(_agentic(), "q")
    assert result == ("", [])
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_agentic_block_renders_when_manager_gate_is_open():
    with (
        patch.object(
            get_session_manager_module,
            "get_session_manager",
            return_value=_manager(),
        ),
        patch(
            "cognee.infrastructure.session.session_context_builder.build_active_context_block",
            new=AsyncMock(return_value=("BLOCK", ["e1"])),
        ) as build,
    ):
        result = await AgenticRetriever._maybe_active_context_block(_agentic(), "q")
    assert result == ("BLOCK", ["e1"])
    build.assert_awaited_once()
