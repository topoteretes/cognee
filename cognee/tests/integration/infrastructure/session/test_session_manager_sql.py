"""Integration tests for SessionManager with SqlCacheAdapter (aiosqlite-backed)."""

import asyncio
import tempfile

import pytest
from sqlalchemy import select
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.cache.sql.tables import cache_qa_entries
from cognee.infrastructure.session.feedback_models import (
    AgentTraceFeedbackSummary,
    FeedbackDetectionResult,
)
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.infrastructure.session.session_turn import SessionTurnPreparation


@pytest.fixture
def sql_adapter():
    """SqlCacheAdapter backed by a temporary aiosqlite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmpdir}/cache.db")
        yield adapter
        asyncio.run(adapter.close())


@pytest.fixture
def session_manager(sql_adapter):
    """SessionManager wired to SqlCacheAdapter."""
    return SessionManager(cache_engine=sql_adapter)


async def _qa_expirations(adapter, user_id: str, session_id: str):
    """Read expires_at for every QA row of a session directly from the table."""
    async with adapter.sessionmaker() as session:
        result = await session.execute(
            select(cache_qa_entries.c.expires_at).where(
                cache_qa_entries.c.user_id == user_id,
                cache_qa_entries.c.session_id == session_id,
            )
        )
        return list(result.scalars().all())


@pytest.mark.asyncio
async def test_add_qa_and_get_session(session_manager):
    """Add QA via SessionManager and retrieve via get_session."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )
    assert qa_id is not None

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].question == "Q1?"
    assert entries[0].answer == "A1."
    assert entries[0].qa_id == qa_id


@pytest.mark.asyncio
async def test_add_qa_sets_session_ttl(session_manager, sql_adapter):
    """Session writes through SessionManager stamp expires_at on the session rows."""
    await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )

    expirations = await _qa_expirations(sql_adapter, "u1", "s1")
    assert len(expirations) == 1
    assert expirations[0] is not None


@pytest.mark.asyncio
async def test_get_session_does_not_refresh_session_ttl(session_manager, sql_adapter):
    """Read-only session access should not refresh expires_at."""
    await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )
    before = await _qa_expirations(sql_adapter, "u1", "s1")

    entries = await session_manager.get_session(user_id="u1", session_id="s1")

    after = await _qa_expirations(sql_adapter, "u1", "s1")
    assert len(entries) == 1
    assert before == after


@pytest.mark.asyncio
async def test_add_agent_trace_step_and_get_trace_session(session_manager):
    """Trace steps appended via SessionManager are returned in append order."""
    with (
        patch(
            "cognee.infrastructure.session.session_agent_trace.read_query_prompt",
            return_value="summarize this",
        ),
        patch(
            "cognee.infrastructure.session.session_agent_trace.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
            return_value=AgentTraceFeedbackSummary(session_feedback="Plan created successfully."),
        ),
    ):
        trace_id_1 = await session_manager.add_agent_trace_step(
            user_id="u1",
            session_id="s1",
            origin_function="plan_trip",
            status="success",
            memory_query="trip preferences",
            memory_context="User likes quiet places",
            method_params={"city": "Tokyo"},
            method_return_value="Plan created",
        )
        trace_id_2 = await session_manager.add_agent_trace_step(
            user_id="u1",
            session_id="s1",
            origin_function="book_hotel",
            status="error",
            method_params={"area": "Shibuya"},
            error_message="No availability",
        )

    entries = await session_manager.get_agent_trace_session(user_id="u1", session_id="s1")
    feedback = await session_manager.get_agent_trace_feedback(user_id="u1", session_id="s1")

    assert [entry.trace_id for entry in entries] == [trace_id_1, trace_id_2]
    assert entries[0].origin_function == "plan_trip"
    assert entries[1].origin_function == "book_hotel"
    assert feedback == [
        "Plan created successfully.",
        "book_hotel failed. Reason: No availability.",
    ]


@pytest.mark.asyncio
async def test_add_agent_trace_step_can_disable_llm_feedback_generation(session_manager):
    """Disabling LLM feedback generation stores deterministic fallback feedback."""
    with patch(
        "cognee.infrastructure.session.session_agent_trace.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
    ) as mock_llm:
        trace_id = await session_manager.add_agent_trace_step(
            user_id="u1",
            session_id="s1",
            origin_function="plan_trip",
            status="success",
            method_return_value="Plan created",
            generate_feedback_with_llm=False,
        )

    assert trace_id is not None
    mock_llm.assert_not_awaited()
    entries = await session_manager.get_agent_trace_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].session_feedback == "plan_trip succeeded."


@pytest.mark.asyncio
async def test_agent_trace_session_isolated_by_user_and_session(session_manager):
    """Agent trace sessions remain isolated by user_id and session_id."""
    await session_manager.add_agent_trace_step(
        user_id="u1",
        session_id="s1",
        origin_function="plan_trip",
        status="success",
    )
    await session_manager.add_agent_trace_step(
        user_id="u1",
        session_id="s2",
        origin_function="book_hotel",
        status="error",
        error_message="No availability",
    )
    await session_manager.add_agent_trace_step(
        user_id="u2",
        session_id="s1",
        origin_function="book_flight",
        status="success",
    )

    u1s1 = await session_manager.get_agent_trace_session(user_id="u1", session_id="s1")
    u1s2 = await session_manager.get_agent_trace_session(user_id="u1", session_id="s2")
    u2s1 = await session_manager.get_agent_trace_session(user_id="u2", session_id="s1")

    assert len(u1s1) == 1
    assert u1s1[0].origin_function == "plan_trip"
    assert len(u1s2) == 1
    assert u1s2[0].origin_function == "book_hotel"
    assert len(u2s1) == 1
    assert u2s1[0].origin_function == "book_flight"


@pytest.mark.asyncio
async def test_add_qa_with_used_graph_element_ids_round_trip(session_manager):
    """add_qa with used_graph_element_ids stores and returns it via get_session."""
    used_ids = {"node_ids": ["n1"], "edge_ids": ["e1"]}
    qa_id = await session_manager.add_qa(
        user_id="u1",
        question="Q?",
        context="C",
        answer="A",
        session_id="s1",
        used_graph_element_ids=used_ids,
    )
    assert qa_id is not None
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].used_graph_element_ids == used_ids


@pytest.mark.asyncio
async def test_get_session_formatted(session_manager):
    """get_session with formatted=True returns prompt string."""
    await session_manager.add_qa(
        user_id="u1", question="Q?", context="C", answer="A", session_id="s1"
    )
    formatted = await session_manager.get_session(user_id="u1", formatted=True, session_id="s1")
    assert isinstance(formatted, str)
    assert "Previous conversation" in formatted and "Q?" in formatted


@pytest.mark.asyncio
async def test_update_qa(session_manager):
    """update_qa updates entry via SqlCacheAdapter."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.update_qa(
        user_id="u1", qa_id=qa_id, question="Q updated?", session_id="s1"
    )
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0].question == "Q updated?"


@pytest.mark.asyncio
async def test_add_feedback(session_manager):
    """add_feedback sets feedback on entry."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.add_feedback(
        user_id="u1", qa_id=qa_id, feedback_score=5, session_id="s1"
    )
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0].feedback_score == 5


@pytest.mark.asyncio
async def test_delete_feedback(session_manager):
    """delete_feedback clears feedback."""
    qa_id = await session_manager.add_qa(
        user_id="u1",
        question="Q",
        context="C",
        answer="A",
        session_id="s1",
        feedback_text="good",
        feedback_score=4,
    )
    ok = await session_manager.delete_feedback(user_id="u1", qa_id=qa_id, session_id="s1")
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0].feedback_score is None
    assert entries[0].feedback_text is None


@pytest.mark.asyncio
async def test_delete_qa(session_manager):
    """delete_qa removes single entry."""
    qa1 = await session_manager.add_qa(
        user_id="u1", question="Q1", context="C1", answer="A1", session_id="s1"
    )
    await session_manager.add_qa(
        user_id="u1", question="Q2", context="C2", answer="A2", session_id="s1"
    )
    ok = await session_manager.delete_qa(user_id="u1", qa_id=qa1, session_id="s1")
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].question == "Q2"


@pytest.mark.asyncio
async def test_delete_session(session_manager):
    """delete_session clears both QA and trace session entries."""
    await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    trace_id = await session_manager.add_agent_trace_step(
        user_id="u1",
        session_id="s1",
        origin_function="plan_trip",
        status="success",
    )
    assert trace_id is not None
    ok = await session_manager.delete_session(user_id="u1", session_id="s1")
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries == []
    trace_entries = await session_manager.get_agent_trace_session(user_id="u1", session_id="s1")
    assert trace_entries == []


@pytest.mark.asyncio
async def test_graph_context_round_trip_via_kv_interface(session_manager, sql_adapter):
    """set/get_graph_context round-trips through the adapter's KV methods (cache_kv table)."""
    await session_manager.set_graph_context(
        user_id="u1", session_id="s1", context="graph snapshot text"
    )

    # Stored under the verbatim legacy key string, readable via the interface method.
    assert await sql_adapter.get_value("graph_knowledge:u1:s1") == "graph snapshot text"
    assert (
        await session_manager.get_graph_context(user_id="u1", session_id="s1")
        == "graph snapshot text"
    )

    # delete_session also removes the graph knowledge snapshot key.
    await session_manager.delete_session(user_id="u1", session_id="s1")
    assert await sql_adapter.get_value("graph_knowledge:u1:s1") is None
    assert await session_manager.get_graph_context(user_id="u1", session_id="s1") == ""


@pytest.mark.asyncio
async def test_get_graph_context_returns_empty_when_unset(session_manager):
    """get_graph_context returns empty string when no snapshot was stored."""
    assert await session_manager.get_graph_context(user_id="u1", session_id="missing") == ""


@pytest.mark.asyncio
async def test_generate_completion_with_session_saves_qa(session_manager):
    """generate_completion_with_session runs completion and saves QA to session (LLM mocked)."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    with (
        patch("cognee.infrastructure.session.session_manager.session_user") as mock_session_user,
        patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
        patch(
            "cognee.infrastructure.session.session_turn.generate_session_completion_with_optional_summary",
            new_callable=AsyncMock,
            return_value=("Integration test answer", "", None),
        ),
    ):
        mock_session_user.get.return_value = mock_user
        mock_config = MagicMock()
        mock_config.caching = True
        mock_config_cls.return_value = mock_config

        used_ids = {"node_ids": ["n1"]}
        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="What is X?",
            context="Context about X.",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
            used_graph_element_ids=used_ids,
            turn_preparation=SessionTurnPreparation(
                should_answer=True,
                effective_query="What is X?",
            ),
        )

    assert result == "Integration test answer"
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].question == "What is X?"
    assert entries[0].answer == "Integration test answer"
    assert entries[0].used_graph_element_ids == used_ids


@pytest.mark.asyncio
async def test_generate_completion_with_session_feedback_only_records_acknowledgement_qa(
    session_manager,
):
    """When no query_to_answer is detected: acknowledgement returned and stored.

    Feedback routing now happens pre-completion via analyze_turn_for_session_context. When the
    analysis yields no query_to_answer (feedback only) and a previous answer exists, completion is
    skipped but the acknowledgement turn remains recallable in the session history.
    """
    qa_id = await session_manager.add_qa(
        user_id="u1",
        question="What is X?",
        context="Context about X.",
        answer="X is something.",
        session_id="s1",
    )
    assert qa_id is not None

    mock_user = MagicMock()
    mock_user.id = "u1"
    with (
        patch("cognee.infrastructure.session.session_manager.session_user") as mock_session_user,
        patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
        patch(
            "cognee.infrastructure.session.session_turn.analyze_turn_for_session_context",
            new_callable=AsyncMock,
            return_value=FeedbackDetectionResult(response_to_user="Thanks for your feedback!"),
        ),
        patch(
            "cognee.infrastructure.session.session_turn.generate_session_completion_with_optional_summary",
            new_callable=AsyncMock,
            return_value=("Generated answer", "", None),
        ) as mock_generate,
    ):
        mock_session_user.get.return_value = mock_user
        mock_config = MagicMock()
        mock_config.caching = True
        mock_config.auto_feedback = True
        mock_config_cls.return_value = mock_config

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="thanks, that was helpful!",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Thanks for your feedback!"
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 2
    assert entries[0].qa_id == qa_id
    assert entries[0].question == "What is X?"
    assert entries[1].question == "thanks, that was helpful!"
    assert entries[1].answer == "Thanks for your feedback!"
    mock_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_completion_with_session_feedback_and_followup_adds_qa(session_manager):
    """When query_to_answer is present: new QA is added with answer.

    When the analysis carries a query_to_answer, the turn is answered using that effective query
    and the answer is persisted as a new QA.
    """
    qa_id_first = await session_manager.add_qa(
        user_id="u1",
        question="What is X?",
        context="Context about X.",
        answer="X is something.",
        session_id="s1",
    )
    assert qa_id_first is not None

    mock_user = MagicMock()
    mock_user.id = "u1"
    with (
        patch("cognee.infrastructure.session.session_manager.session_user") as mock_session_user,
        patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
        patch(
            "cognee.infrastructure.session.session_turn.analyze_turn_for_session_context",
            new_callable=AsyncMock,
            return_value=FeedbackDetectionResult(
                response_to_user="Thanks for your feedback!",
                query_to_answer="What is the capital of France?",
            ),
        ),
        patch(
            "cognee.infrastructure.session.session_turn.generate_session_completion_with_optional_summary",
            new_callable=AsyncMock,
            return_value=("Paris is the capital of France.", "", None),
        ),
    ):
        mock_session_user.get.return_value = mock_user
        mock_config = MagicMock()
        mock_config.caching = True
        mock_config.auto_feedback = True
        mock_config_cls.return_value = mock_config

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="thanks! What is the capital of France?",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Paris is the capital of France."
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 2
    first_qa = next((e for e in entries if e.qa_id == qa_id_first), None)
    followup_qa = next(
        (e for e in entries if e.question == "thanks! What is the capital of France?"),
        None,
    )
    assert first_qa is not None
    assert first_qa.feedback_text is None
    assert first_qa.feedback_score is None
    assert followup_qa is not None
    assert followup_qa.answer == "Paris is the capital of France."
