"""One session_id over several datasets is one turn (SharedSessionTurn).

The datasets of one search share the turn's context load and analysis, apply the
analysis once, record one QA row each, and do not take the turn lock themselves (the
search fan-out holds it for the whole search).
"""

import asyncio
from contextlib import ExitStack, asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_concurrent_turn import (
    SessionTurnContext,
    SharedSessionTurn,
    shared_session_turn,
)
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.session_aware_completion import run_concurrent_session_turn

MODULE = "cognee.modules.retrieval.session_aware_completion"
USER_ID = uuid4()


def _session_manager():
    manager = MagicMock()
    manager.resolve_session_id.side_effect = lambda session_id: session_id
    manager.is_auto_feedback_enabled.return_value = True
    return manager


def _retriever():
    retriever = CompletionRetriever(session_id="s1")
    retriever.extract_context_object_ids = lambda objects: {}
    retriever.append_references = AsyncMock(side_effect=lambda answers, _objects: answers)
    return retriever


@asynccontextmanager
async def _forbidden_lock(*_args, **_kwargs):
    raise AssertionError("a dataset of a shared turn must not take the turn lock")
    yield


@contextmanager
def _patched(manager, *, load, analyze, lock):
    async def retrieve_and_answer(_retriever, **_kwargs):
        await asyncio.sleep(0)
        return ["chunk"], "context", "answer"

    patchers = (
        patch(f"{MODULE}.session_user", SimpleNamespace(get=lambda: SimpleNamespace(id=USER_ID))),
        patch(f"{MODULE}.get_session_manager", return_value=manager),
        patch(f"{MODULE}.load_turn_context", load),
        patch(f"{MODULE}.analyze_turn", analyze),
        patch(f"{MODULE}._retrieve_and_answer", retrieve_and_answer),
        patch(f"{MODULE}.session_turn_lock", lock),
    )
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield


@pytest.mark.asyncio
async def test_datasets_of_one_search_share_one_turn():
    load = AsyncMock(return_value=SessionTurnContext(raw_message="question"))
    analyze = AsyncMock(return_value=SessionTurnAnalysis())
    with (
        _patched(_session_manager(), load=load, analyze=analyze, lock=_forbidden_lock),
        patch(f"{MODULE}.commit_turn", new_callable=AsyncMock) as commit,
    ):
        token = shared_session_turn.set(SharedSessionTurn(user_id=str(USER_ID), session_id="s1"))
        try:
            results = await asyncio.gather(
                *(run_concurrent_session_turn(_retriever(), raw_query="question") for _ in range(3))
            )
        finally:
            shared_session_turn.reset(token)

    assert [completion for _objects, _context, completion in results] == [["answer"]] * 3
    load.assert_awaited_once()
    analyze.assert_awaited_once()
    # One QA row per dataset, but the analysis is applied to the session only once.
    assert commit.await_count == 3
    applied = [call.kwargs["apply_analysis"] for call in commit.await_args_list]
    assert sorted(applied) == [False, False, True]


@pytest.mark.asyncio
async def test_a_turn_outside_a_shared_search_still_takes_the_lock():
    load = AsyncMock(return_value=SessionTurnContext(raw_message="question"))
    analyze = AsyncMock(return_value=SessionTurnAnalysis())
    entered = []

    @asynccontextmanager
    async def recording_lock(user_id, session_id):
        entered.append((user_id, session_id))
        yield

    with (
        _patched(_session_manager(), load=load, analyze=analyze, lock=recording_lock),
        patch(f"{MODULE}.commit_turn", new_callable=AsyncMock) as commit,
    ):
        await run_concurrent_session_turn(_retriever(), raw_query="question")

    assert entered == [(str(USER_ID), "s1")]
    assert commit.await_args.kwargs["apply_analysis"] is True


@pytest.mark.asyncio
async def test_a_shared_turn_for_another_session_is_ignored():
    load = AsyncMock(return_value=SessionTurnContext(raw_message="question"))
    analyze = AsyncMock(return_value=SessionTurnAnalysis())
    entered = []

    @asynccontextmanager
    async def recording_lock(user_id, session_id):
        entered.append((user_id, session_id))
        yield

    with (
        _patched(_session_manager(), load=load, analyze=analyze, lock=recording_lock),
        patch(f"{MODULE}.commit_turn", new_callable=AsyncMock),
    ):
        token = shared_session_turn.set(
            SharedSessionTurn(user_id=str(USER_ID), session_id="another-session")
        )
        try:
            await run_concurrent_session_turn(_retriever(), raw_query="question")
        finally:
            shared_session_turn.reset(token)

    assert entered == [(str(USER_ID), "s1")]


@pytest.mark.asyncio
async def test_shared_turn_starts_the_analysis_once_for_concurrent_callers():
    turn = SharedSessionTurn(user_id="u", session_id="s")
    calls = []

    async def analyze():
        calls.append(1)
        await asyncio.sleep(0)
        return SessionTurnAnalysis()

    results = await asyncio.gather(*(turn.analysis(analyze) for _ in range(5)))

    assert len(calls) == 1
    assert all(result is results[0] for result in results)
    assert [turn.claim_analysis_application() for _ in range(3)] == [True, False, False]
