"""Integration-style tests for a concurrent session turn against a REAL cache adapter.

The unit tests for `session_concurrent_turn` fake the session manager entirely, so nothing
there would notice if `commit_turn`'s `add_qa` call stopped matching the real signature —
`commit_turn` fails open on that write by design, so a mismatch would be silent in both
production and tests. These exercise the write-then-read round trip against an FS-backed
cache, with only the vector index and the LLM mocked.

Mirrors `test_session_context_turn_flow.py`, which does the same for the sequential path.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_concurrent_turn import (
    SessionTurnContext,
    commit_turn,
    load_turn_context,
)
from cognee.infrastructure.session.session_manager import SessionManager

USER_ID = "owner-1"
SESSION_ID = "s1"


@pytest.fixture
def fs_adapter():
    """FSCacheAdapter backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ):
            from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                FSCacheAdapter,
            )

            inst = FSCacheAdapter()
            yield inst
            inst.cache.close()


@pytest.fixture
def session_manager(fs_adapter) -> SessionManager:
    return SessionManager(cache_engine=fs_adapter)


@pytest.fixture(autouse=True)
def session_mocks():
    """Cache is real; the vector index and the config are not."""
    cfg = MagicMock()
    cfg.caching = True
    cfg.auto_feedback = True
    cfg.max_session_context_chars = None

    with (
        patch(
            "cognee.infrastructure.session.session_manager.index_session_qa", new_callable=AsyncMock
        ),
        patch(
            "cognee.infrastructure.session.session_turn.search_session_qa_ids",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("cognee.infrastructure.session.session_manager.get_cache_config", return_value=cfg),
    ):
        yield


async def _commit(session_manager, *, question, answer, served_ids=(), graph_ids=None):
    """Commit one turn exactly the way `run_concurrent_session_turn` does."""
    await commit_turn(
        session_manager,
        snapshot=SessionTurnContext(raw_message=question, active_context_ids=tuple(served_ids)),
        analysis=SessionTurnAnalysis(),
        answer=answer,
        user_id=USER_ID,
        session_id=SESSION_ID,
        used_graph_element_ids=graph_ids,
    )


@pytest.mark.asyncio
async def test_committed_turn_is_read_back_by_the_next_turn(session_manager):
    """commit_turn -> load_turn_context: the next turn sees the one before it."""
    await _commit(
        session_manager,
        question="what is the capital of France?",
        answer="Paris.",
        served_ids=("ctx-1", "ctx-2"),
        graph_ids={"node_ids": ["n1"], "edge_ids": ["e1"]},
    )

    turn_context = await load_turn_context(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        raw_message="and its population?",
    )

    assert len(turn_context.recent_qas) == 1
    _qa_id, question, answer = turn_context.recent_qas[0]
    assert (question, answer) == ("what is the capital of France?", "Paris.")
    assert turn_context.previous_qa_id is not None
    assert turn_context.previous_question == "what is the capital of France?"
    assert turn_context.previous_answer == "Paris."
    assert "Paris." in turn_context.completion_history


@pytest.mark.asyncio
async def test_commit_persists_the_ids_add_qa_is_given(session_manager):
    """The kwargs commit_turn passes must survive a real adapter round trip.

    commit_turn swallows add_qa failures, so without this a signature drift would be
    invisible: the turn would still answer and the QA row would silently never appear.
    """
    await _commit(
        session_manager,
        question="question",
        answer="answer",
        served_ids=("ctx-1", "ctx-2"),
        graph_ids={"node_ids": ["n1"], "edge_ids": ["e1"]},
    )

    entries = await session_manager.get_session(
        user_id=USER_ID, session_id=SESSION_ID, formatted=False
    )

    assert len(entries) == 1
    assert entries[0].used_session_context_ids == ["ctx-1", "ctx-2"]
    assert entries[0].used_graph_element_ids == {"node_ids": ["n1"], "edge_ids": ["e1"]}
    # Concurrent turns store no context, matching the sequential path for these retrievers.
    assert entries[0].context == ""


@pytest.mark.asyncio
async def test_recent_qas_are_capped_at_two_and_ordered_oldest_first(session_manager):
    """build_conversational_query relies on this bound and this order."""
    for index in range(3):
        await _commit(session_manager, question=f"question {index}", answer=f"answer {index}")

    turn_context = await load_turn_context(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        raw_message="next",
    )

    assert [question for _id, question, _answer in turn_context.recent_qas] == [
        "question 1",
        "question 2",
    ]
    assert turn_context.previous_question == "question 2"


@pytest.mark.asyncio
async def test_first_turn_reads_an_empty_session(session_manager):
    turn_context = await load_turn_context(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        raw_message="first question",
    )

    assert turn_context.raw_message == "first question"
    assert turn_context.recent_qas == ()
    assert turn_context.previous_qa_id is None
    assert turn_context.previous_served_context == ()
