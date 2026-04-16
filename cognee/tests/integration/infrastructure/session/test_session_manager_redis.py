"""Integration tests for SessionManager with RedisAdapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.infrastructure.session.feedback_models import (
    AgentTraceFeedbackSummary,
    FeedbackDetectionResult,
)
from cognee.infrastructure.session.session_manager import SessionManager


class _InMemoryRedisList:
    """Minimal in-memory Redis list emulation."""

    def __init__(self):
        self.data: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def rpush(self, key: str, *vals: str):
        self.data.setdefault(key, []).extend(vals)

    async def lrange(self, key: str, start: int, end: int):
        lst = self.data.get(key, [])
        s = start if start >= 0 else len(lst) + start
        e = (end + 1) if end >= 0 else len(lst) + end + 1
        return lst[s:e]

    async def lindex(self, key: str, idx: int):
        lst = self.data.get(key, [])
        return lst[idx] if -len(lst) <= idx < len(lst) else None

    async def lset(self, key: str, idx: int, val: str):
        self.data[key][idx] = val

    async def delete(self, key: str):
        self.ttls.pop(key, None)
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl
        self.expire_calls.append((key, ttl))

    async def ttl(self, key: str):
        if key not in self.data:
            return -2
        return self.ttls.get(key, -1)

    async def flushdb(self):
        self.data.clear()
        self.ttls.clear()
        self.expire_calls.clear()


@pytest.fixture
def redis_adapter():
    """RedisAdapter with in-memory backend."""
    store = _InMemoryRedisList()
    patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_mod}.aioredis.Redis", return_value=store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        yield RedisAdapter(host="localhost", port=6379)


@pytest.fixture
def session_manager(redis_adapter):
    """SessionManager wired to RedisAdapter."""
    return SessionManager(cache_engine=redis_adapter)


@pytest.mark.asyncio
async def test_add_qa_and_get_session(session_manager):
    """Add QA via SessionManager and retrieve via get_session."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )
    assert qa_id is not None

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "Q1?"
    assert entries[0]["answer"] == "A1."
    assert entries[0]["qa_id"] == qa_id


@pytest.mark.asyncio
async def test_add_qa_sets_session_ttl(session_manager, redis_adapter):
    """Session writes through SessionManager apply Redis TTL to the session key."""
    await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )

    assert await redis_adapter.async_redis.ttl("agent_sessions:u1:s1") == 604800


@pytest.mark.asyncio
async def test_get_session_does_not_refresh_session_ttl(session_manager, redis_adapter):
    """Read-only session access should not refresh TTL."""
    await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )
    redis_adapter.async_redis.expire_calls.clear()

    entries = await session_manager.get_session(user_id="u1", session_id="s1")

    assert len(entries) == 1
    assert redis_adapter.async_redis.expire_calls == []


@pytest.mark.asyncio
async def test_add_agent_trace_step_and_get_trace_session(session_manager):
    """Trace steps appended via SessionManager are returned in append order."""
    with (
        patch(
            "cognee.infrastructure.session.session_manager.read_query_prompt",
            return_value="summarize this",
        ),
        patch(
            "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
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

    assert [entry["trace_id"] for entry in entries] == [trace_id_1, trace_id_2]
    assert entries[0]["origin_function"] == "plan_trip"
    assert entries[1]["origin_function"] == "book_hotel"
    assert feedback == [
        "Plan created successfully.",
        "book_hotel failed. Reason: No availability.",
    ]


@pytest.mark.asyncio
async def test_add_agent_trace_step_can_disable_llm_feedback_generation(session_manager):
    """Disabling LLM feedback generation stores deterministic fallback feedback."""
    with patch(
        "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
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
    assert entries[0]["session_feedback"] == "plan_trip succeeded."


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
    assert u1s1[0]["origin_function"] == "plan_trip"
    assert len(u1s2) == 1
    assert u1s2[0]["origin_function"] == "book_hotel"
    assert len(u2s1) == 1
    assert u2s1[0]["origin_function"] == "book_flight"


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
    assert entries[0]["used_graph_element_ids"] == used_ids


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
    """update_qa updates entry via RedisAdapter."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.update_qa(
        user_id="u1", qa_id=qa_id, question="Q updated?", session_id="s1"
    )
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0]["question"] == "Q updated?"


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
    assert entries[0]["feedback_score"] == 5


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
    assert entries[0].get("feedback_score") is None
    assert entries[0].get("feedback_text") is None


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
    assert entries[0]["question"] == "Q2"


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
async def test_generate_completion_with_session_saves_qa(session_manager):
    """generate_completion_with_session runs completion and saves QA to session (LLM mocked)."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    with (
        patch("cognee.infrastructure.session.session_manager.session_user") as mock_session_user,
        patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
        patch(
            "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
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
        )

    assert result == "Integration test answer"
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "What is X?"
    assert entries[0]["answer"] == "Integration test answer"
    assert entries[0]["used_graph_element_ids"] == used_ids


@pytest.mark.asyncio
async def test_generate_completion_with_session_feedback_only_no_new_qa(session_manager):
    """When feedback only is detected: feedback persisted on last QA, no new QA added."""
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
            "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
            new_callable=AsyncMock,
            return_value=(
                "Generated answer",
                "",
                FeedbackDetectionResult(
                    feedback_detected=True,
                    feedback_text="User said thanks.",
                    feedback_score=5.0,
                    response_to_user="Thanks for your feedback!",
                    contains_followup_question=False,
                ),
            ),
        ),
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
    assert len(entries) == 1
    assert entries[0]["qa_id"] == qa_id
    assert entries[0]["question"] == "What is X?"
    assert entries[0].get("feedback_text") == "User said thanks."
    assert entries[0].get("feedback_score") == 5


@pytest.mark.asyncio
async def test_generate_completion_with_session_feedback_and_followup_adds_qa(session_manager):
    """When feedback + follow-up: feedback on last QA and new QA added with answer."""
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
            "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
            new_callable=AsyncMock,
            return_value=(
                "Paris is the capital of France.",
                "",
                FeedbackDetectionResult(
                    feedback_detected=True,
                    feedback_text="User gave thanks and asked follow-up.",
                    feedback_score=5.0,
                    response_to_user="Thanks for your feedback!",
                    contains_followup_question=True,
                ),
            ),
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

    assert "Thanks for your feedback!" in result
    assert "Paris is the capital of France." in result
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 2
    first_qa = next((e for e in entries if e.get("qa_id") == qa_id_first), None)
    followup_qa = next(
        (e for e in entries if e.get("question") == "thanks! What is the capital of France?"),
        None,
    )
    assert first_qa is not None
    assert first_qa.get("feedback_text") == "User gave thanks and asked follow-up."
    assert first_qa.get("feedback_score") == 5
    assert followup_qa is not None
    assert followup_qa["answer"] == "Paris is the capital of France."
