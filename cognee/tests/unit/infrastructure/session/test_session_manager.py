import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.databases.exceptions import SessionParameterValidationError
from cognee.infrastructure.session.feedback_models import (
    AgentTraceFeedbackSummary,
    FeedbackDetectionResult,
)
from cognee.infrastructure.session.session_manager import (
    SessionManager,
    _validate_session_params,
)


class TestValidateSessionParams:
    """Tests for _validate_session_params."""

    def test_valid_params(self):
        """Valid user_id and session_id do not raise."""
        _validate_session_params(user_id="u1", session_id="s1")
        _validate_session_params(user_id="u1", session_id="s1", qa_id="q1")

    def test_empty_user_id_raises(self):
        """Empty user_id raises SessionParameterValidationError."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="", session_id="s1")
        assert "user_id" in exc_info.value.message

    def test_empty_session_id_raises(self):
        """Empty session_id raises SessionParameterValidationError."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="")
        assert "session_id" in exc_info.value.message

    def test_whitespace_user_id_raises(self):
        """Whitespace-only user_id raises."""
        with pytest.raises(SessionParameterValidationError):
            _validate_session_params(user_id="  ", session_id="s1")

    def test_empty_qa_id_raises(self):
        """Empty qa_id raises when provided."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="s1", qa_id="")
        assert "qa_id" in exc_info.value.message

    def test_valid_last_n(self):
        """Valid last_n (positive int or None) does not raise."""
        _validate_session_params(user_id="u1", session_id="s1", last_n=5)
        _validate_session_params(user_id="u1", session_id="s1", last_n=1)

    def test_invalid_last_n_zero_raises(self):
        """last_n=0 raises SessionParameterValidationError."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="s1", last_n=0)
        assert "last_n" in exc_info.value.message

    def test_invalid_last_n_negative_raises(self):
        """last_n negative raises."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="s1", last_n=-1)
        assert "last_n" in exc_info.value.message

    def test_invalid_last_n_not_int_raises(self):
        """last_n not an int raises."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="s1", last_n="5")
        assert "last_n" in exc_info.value.message


class TestSessionManager:
    """Unit tests for SessionManager with mocked cache."""

    @pytest.fixture
    def mock_cache(self):
        """Mock cache engine."""
        cache = MagicMock()
        cache.create_qa_entry = AsyncMock()
        cache.get_all_qa_entries = AsyncMock(return_value=[])
        cache.get_latest_qa_entries = AsyncMock(return_value=[])
        cache.append_agent_trace_step = AsyncMock()
        cache.get_agent_trace_session = AsyncMock(return_value=[])
        cache.get_agent_trace_feedback = AsyncMock(return_value=[])
        cache.get_agent_trace_count = AsyncMock(return_value=0)
        cache.update_qa_entry = AsyncMock(return_value=True)
        cache.delete_feedback = AsyncMock(return_value=True)
        cache.delete_qa_entry = AsyncMock(return_value=True)
        cache.delete_session = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def sm(self, mock_cache):
        """SessionManager with mocked cache."""
        return SessionManager(cache_engine=mock_cache)

    @pytest.fixture
    def sm_unavailable(self):
        """SessionManager with no cache."""
        return SessionManager(cache_engine=None)

    def test_is_available(self, sm, sm_unavailable):
        """is_available reflects cache presence."""
        assert sm.is_available is True
        assert sm_unavailable.is_available is False

    @pytest.mark.asyncio
    async def test_add_qa_session_id_none_uses_default(self, sm, mock_cache):
        """add_qa with session_id=None uses default_session_id."""
        qa_id = await sm.add_qa(user_id="u1", question="Q", context="C", answer="A")
        assert qa_id is not None
        call_kw = mock_cache.create_qa_entry.call_args.kwargs
        assert call_kw["session_id"] == "default_session"

    @pytest.mark.asyncio
    async def test_add_qa_returns_qa_id(self, sm, mock_cache):
        """add_qa returns generated qa_id and calls cache."""
        used_ids = {"node_ids": ["n1"], "edge_ids": ["e1"]}
        qa_id = await sm.add_qa(
            user_id="u1",
            question="Q",
            context="C",
            answer="A",
            session_id="s1",
            used_graph_element_ids=used_ids,
        )
        assert qa_id is not None
        mock_cache.create_qa_entry.assert_called_once()
        call_kw = mock_cache.create_qa_entry.call_args.kwargs
        assert call_kw["user_id"] == "u1"
        assert call_kw["session_id"] == "s1"
        assert call_kw["question"] == "Q"
        assert call_kw["answer"] == "A"
        assert call_kw["qa_id"] == qa_id
        assert call_kw["used_graph_element_ids"] == used_ids

    @pytest.mark.asyncio
    async def test_add_qa_unavailable_returns_none(self, sm_unavailable):
        """add_qa returns None when cache unavailable."""
        assert (
            await sm_unavailable.add_qa(
                user_id="u1", question="Q", context="C", answer="A", session_id="s1"
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_add_qa_invalid_params_raises(self, sm):
        """add_qa raises on invalid user_id or session_id."""
        with pytest.raises(SessionParameterValidationError):
            await sm.add_qa(user_id="", question="Q", context="C", answer="A", session_id="s1")
        with pytest.raises(SessionParameterValidationError):
            await sm.add_qa(user_id="u1", question="Q", context="C", answer="A", session_id="")

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_session_id_none_uses_default(self, sm, mock_cache):
        """add_agent_trace_step with session_id=None uses default_session_id."""
        trace_id = await sm.add_agent_trace_step(
            user_id="u1",
            origin_function="plan_trip",
            status="success",
        )
        assert trace_id is not None
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_id"] == "default_session"

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_returns_trace_id_and_feedback(self, sm, mock_cache):
        """add_agent_trace_step returns generated trace_id and persists generated feedback."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.read_query_prompt",
                return_value="summarize this",
            ),
            patch(
                "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=AgentTraceFeedbackSummary(
                    session_feedback="Trip plan created successfully."
                ),
            ),
        ):
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="plan_trip",
                status="success",
                session_id="s1",
                memory_query="trip preferences",
                memory_context="User likes quiet places",
                method_params={"city": "Tokyo"},
                method_return_value="Plan created",
            )
        assert trace_id is not None
        mock_cache.append_agent_trace_step.assert_called_once()
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["trace_id"] == trace_id
        assert call_kw["origin_function"] == "plan_trip"
        assert call_kw["status"] == "success"
        assert call_kw["memory_query"] == "trip preferences"
        assert call_kw["memory_context"] == "User likes quiet places"
        assert call_kw["method_params"] == {"city": "Tokyo"}
        assert call_kw["method_return_value"] == "Plan created"
        assert call_kw["session_feedback"] == "Trip plan created successfully."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_falls_back_when_summary_is_empty(self, sm, mock_cache):
        """Empty LLM summaries fall back to the deterministic feedback string."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.read_query_prompt",
                return_value="summarize this",
            ),
            patch(
                "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=AgentTraceFeedbackSummary(session_feedback="   "),
            ),
        ):
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="book_hotel",
                status="error",
                session_id="s1",
                method_return_value={"status": "failed"},
                error_message="No availability",
            )
        assert trace_id is not None
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_feedback"] == "book_hotel failed. Reason: No availability."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_falls_back_when_llm_raises(self, sm, mock_cache):
        """LLM failures do not block trace writes and use deterministic fallback feedback."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.read_query_prompt",
                return_value="summarize this",
            ),
            patch(
                "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                side_effect=RuntimeError("llm unavailable"),
            ),
        ):
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="book_hotel",
                status="error",
                session_id="s1",
                method_return_value={"status": "failed"},
                error_message="No availability",
            )
        assert trace_id is not None
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_feedback"] == "book_hotel failed. Reason: No availability."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_falls_back_when_prompt_missing(self, sm, mock_cache):
        """Missing trace feedback prompt uses deterministic fallback feedback."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.read_query_prompt",
                return_value=None,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="plan_trip",
                status="success",
                session_id="s1",
                method_return_value="Plan created",
            )

        assert trace_id is not None
        mock_llm.assert_not_awaited()
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_feedback"] == "plan_trip succeeded."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_falls_back_when_llm_returns_wrong_type(
        self, sm, mock_cache
    ):
        """Unexpected LLM result types use deterministic fallback feedback."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.read_query_prompt",
                return_value="summarize this",
            ),
            patch(
                "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value="not-a-model",
            ),
        ):
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="plan_trip",
                status="success",
                session_id="s1",
                method_return_value="Plan created",
            )

        assert trace_id is not None
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_feedback"] == "plan_trip succeeded."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_method_return_value_none_uses_fallback_without_llm(
        self, sm, mock_cache
    ):
        """None return values skip LLM generation and use deterministic fallback feedback."""
        with patch(
            "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="plan_trip",
                status="success",
                session_id="s1",
                method_return_value=None,
            )

        assert trace_id is not None
        mock_llm.assert_not_awaited()
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_feedback"] == "plan_trip succeeded."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_can_disable_llm_feedback_generation(self, sm, mock_cache):
        """When disabled explicitly, trace feedback uses fallback without touching the LLM."""
        with patch(
            "cognee.infrastructure.session.session_manager.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            trace_id = await sm.add_agent_trace_step(
                user_id="u1",
                origin_function="plan_trip",
                status="success",
                session_id="s1",
                method_return_value="Plan created",
                generate_feedback_with_llm=False,
            )

        assert trace_id is not None
        mock_llm.assert_not_awaited()
        call_kw = mock_cache.append_agent_trace_step.call_args.kwargs
        assert call_kw["session_feedback"] == "plan_trip succeeded."

    @pytest.mark.asyncio
    async def test_add_agent_trace_step_unavailable_returns_none(self, sm_unavailable):
        """add_agent_trace_step returns None when cache unavailable."""
        result = await sm_unavailable.add_agent_trace_step(
            user_id="u1",
            origin_function="plan_trip",
            status="success",
            session_id="s1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_invalid_last_n_raises(self, sm):
        """get_session raises on invalid last_n."""
        with pytest.raises(SessionParameterValidationError):
            await sm.get_session(user_id="u1", last_n=0, session_id="s1")
        with pytest.raises(SessionParameterValidationError):
            await sm.get_session(user_id="u1", last_n=-1, session_id="s1")

    def test_format_entries_empty(self):
        """format_entries returns empty string for empty list."""
        assert SessionManager.format_entries([]) == ""

    def test_format_entries_formats(self):
        """format_entries produces expected format."""
        entries = [
            {"time": "t1", "question": "Q1", "context": "C1", "answer": "A1"},
        ]
        out = SessionManager.format_entries(entries)
        assert "Previous conversation" in out
        assert "Q1" in out and "A1" in out

    @pytest.mark.asyncio
    async def test_get_session_calls_cache(self, sm, mock_cache):
        """get_session delegates to cache."""
        mock_cache.get_all_qa_entries.return_value = [
            {"qa_id": "1", "question": "Q", "context": "C", "answer": "A", "time": "t"}
        ]
        entries = await sm.get_session(user_id="u1", session_id="s1")
        assert len(entries) == 1
        assert entries[0]["question"] == "Q"
        mock_cache.get_all_qa_entries.assert_called_once_with("u1", "s1")

    @pytest.mark.asyncio
    async def test_get_session_formatted(self, sm, mock_cache):
        """get_session with formatted=True returns string."""
        mock_cache.get_all_qa_entries.return_value = [
            SessionQAEntry(qa_id="1", question="Q", context="C", answer="A", time="t")
        ]
        out = await sm.get_session(user_id="u1", formatted=True, session_id="s1")
        assert isinstance(out, str)
        assert "Previous conversation" in out and "Q" in out

    @pytest.mark.asyncio
    async def test_get_session_unavailable_returns_empty(self, sm_unavailable):
        """get_session returns empty list when cache unavailable."""
        assert await sm_unavailable.get_session(user_id="u1", session_id="s1") == []
        assert await sm_unavailable.get_session(user_id="u1", formatted=True, session_id="s1") == ""

    @pytest.mark.asyncio
    async def test_get_agent_trace_session_calls_cache(self, sm, mock_cache):
        """get_agent_trace_session delegates to cache."""
        mock_cache.get_agent_trace_session.return_value = [
            {
                "trace_id": "t1",
                "origin_function": "plan_trip",
                "status": "success",
                "session_feedback": "plan_trip succeeded.",
            }
        ]
        entries = await sm.get_agent_trace_session(user_id="u1", session_id="s1")
        assert len(entries) == 1
        assert entries[0]["trace_id"] == "t1"
        mock_cache.get_agent_trace_session.assert_called_once_with("u1", "s1", last_n=None)

    @pytest.mark.asyncio
    async def test_get_agent_trace_session_unavailable_returns_empty(self, sm_unavailable):
        """get_agent_trace_session returns empty list when cache unavailable."""
        assert await sm_unavailable.get_agent_trace_session(user_id="u1", session_id="s1") == []

    @pytest.mark.asyncio
    async def test_get_agent_trace_feedback_calls_cache(self, sm, mock_cache):
        """get_agent_trace_feedback delegates to cache and returns feedback only."""
        mock_cache.get_agent_trace_feedback.return_value = [
            "plan_trip succeeded.",
            "book_hotel failed. Reason: No availability.",
        ]
        feedback = await sm.get_agent_trace_feedback(user_id="u1", session_id="s1")
        assert feedback == [
            "plan_trip succeeded.",
            "book_hotel failed. Reason: No availability.",
        ]
        mock_cache.get_agent_trace_feedback.assert_called_once_with("u1", "s1", last_n=None)

    @pytest.mark.asyncio
    async def test_get_agent_trace_feedback_passes_last_n_to_cache(self, sm, mock_cache):
        """get_agent_trace_feedback forwards last_n to cache."""
        mock_cache.get_agent_trace_feedback.return_value = ["book_hotel failed."]

        feedback = await sm.get_agent_trace_feedback(user_id="u1", session_id="s1", last_n=1)

        assert feedback == ["book_hotel failed."]
        mock_cache.get_agent_trace_feedback.assert_called_once_with("u1", "s1", last_n=1)

    @pytest.mark.asyncio
    async def test_get_agent_trace_feedback_unavailable_returns_empty(self, sm_unavailable):
        """get_agent_trace_feedback returns empty list when cache unavailable."""
        assert await sm_unavailable.get_agent_trace_feedback(user_id="u1", session_id="s1") == []

    @pytest.mark.asyncio
    async def test_get_agent_trace_count_calls_cache(self, sm, mock_cache):
        """get_agent_trace_count delegates to cache."""
        mock_cache.get_agent_trace_count.return_value = 3

        count = await sm.get_agent_trace_count(user_id="u1", session_id="s1")

        assert count == 3
        mock_cache.get_agent_trace_count.assert_called_once_with("u1", "s1")

    @pytest.mark.asyncio
    async def test_get_agent_trace_count_unavailable_returns_zero(self, sm_unavailable):
        """get_agent_trace_count returns zero when cache unavailable."""
        assert await sm_unavailable.get_agent_trace_count(user_id="u1", session_id="s1") == 0

    @pytest.mark.asyncio
    async def test_update_qa_calls_cache(self, sm, mock_cache):
        """update_qa delegates to cache."""
        ok = await sm.update_qa(user_id="u1", qa_id="q1", question="Q2", session_id="s1")
        assert ok is True
        mock_cache.update_qa_entry.assert_called_once_with(
            user_id="u1",
            session_id="s1",
            qa_id="q1",
            question="Q2",
            context=None,
            answer=None,
            feedback_text=None,
            feedback_score=None,
            used_graph_element_ids=None,
            memify_metadata=None,
        )

    @pytest.mark.asyncio
    async def test_delete_feedback_calls_cache(self, sm, mock_cache):
        """delete_feedback delegates to cache."""
        ok = await sm.delete_feedback(user_id="u1", qa_id="q1", session_id="s1")
        assert ok is True
        mock_cache.delete_feedback.assert_called_once_with(
            user_id="u1", session_id="s1", qa_id="q1"
        )

    @pytest.mark.asyncio
    async def test_delete_qa_calls_cache(self, sm, mock_cache):
        """delete_qa delegates to cache."""
        ok = await sm.delete_qa(user_id="u1", qa_id="q1", session_id="s1")
        assert ok is True
        mock_cache.delete_qa_entry.assert_called_once_with(
            user_id="u1", session_id="s1", qa_id="q1"
        )

    @pytest.mark.asyncio
    async def test_delete_session_calls_cache(self, sm, mock_cache):
        """delete_session delegates to cache."""
        ok = await sm.delete_session(user_id="u1", session_id="s1")
        assert ok is True
        mock_cache.delete_session.assert_called_once_with(user_id="u1", session_id="s1")

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_no_user_id_calls_generate_completion_only(
        self, sm, mock_cache
    ):
        """When user_id is None, generate_completion_with_session runs completion only, no add_qa."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch(
                "cognee.infrastructure.session.session_manager.generate_completion",
                new_callable=AsyncMock,
                return_value="Generated answer",
            ) as mock_generate,
        ):
            mock_session_user.get.return_value = None

            result = await sm.generate_completion_with_session(
                query="Q?",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Generated answer"
        mock_generate.assert_awaited_once()
        mock_cache.create_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_cache_disabled_calls_generate_completion_only(
        self, sm, mock_cache
    ):
        """When caching disabled, generate_completion_with_session runs completion only, no add_qa."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
            patch(
                "cognee.infrastructure.session.session_manager.generate_completion",
                new_callable=AsyncMock,
                return_value="Generated answer",
            ) as mock_generate,
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = False
            mock_config_cls.return_value = mock_config

            result = await sm.generate_completion_with_session(
                query="Q?",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Generated answer"
        mock_generate.assert_awaited_once()
        mock_cache.create_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_available_calls_add_qa(self, sm, mock_cache):
        """When session available, generate_completion_with_session gets history, generates, saves QA."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
            patch(
                "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
                new_callable=AsyncMock,
                return_value=("Generated answer", "", None),
            ) as mock_generate,
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config_cls.return_value = mock_config

            used_ids = {"node_ids": ["n1"]}
            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="Q?",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
                used_graph_element_ids=used_ids,
            )

        assert result == "Generated answer"
        mock_generate.assert_awaited_once()
        mock_cache.create_qa_entry.assert_called_once()
        call_kw = mock_cache.create_qa_entry.call_args.kwargs
        assert call_kw["user_id"] == "u1"
        assert call_kw["session_id"] == "s1"
        assert call_kw["question"] == "Q?"
        assert call_kw["answer"] == "Generated answer"
        assert call_kw["context"] == ""
        assert call_kw["used_graph_element_ids"] == used_ids

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_unavailable_returns_completion_only(
        self, sm_unavailable
    ):
        """When cache unavailable, generate_completion_with_session runs completion only."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch(
                "cognee.infrastructure.session.session_manager.generate_completion",
                new_callable=AsyncMock,
                return_value="Generated answer",
            ) as mock_generate,
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user

            result = await sm_unavailable.generate_completion_with_session(
                query="Q?",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Generated answer"
        mock_generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_feedback_only_returns_thanks_skips_add_qa(
        self, sm, mock_cache
    ):
        """When feedback detected and no follow-up question: persist feedback, return thanks, skip add_qa."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
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
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = True
            mock_config_cls.return_value = mock_config
            mock_cache.get_latest_qa_entries.return_value = [
                SessionQAEntry(qa_id="last-qa-123", question="Q", context="", answer="A", time="t")
            ]

            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="thanks, that was helpful!",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Thanks for your feedback!"
        mock_cache.update_qa_entry.assert_called_once()
        call_kw = mock_cache.update_qa_entry.call_args.kwargs
        assert call_kw["qa_id"] == "last-qa-123"
        assert call_kw["feedback_text"] == "User said thanks."
        assert call_kw["feedback_score"] == 5
        mock_cache.create_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_feedback_and_followup_persists_and_adds_qa(
        self, sm, mock_cache
    ):
        """When feedback and follow-up question: persist feedback, add_qa, return thanks + completion."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
            patch(
                "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
                new_callable=AsyncMock,
                return_value=(
                    "Paris is the capital of France.",
                    "",
                    FeedbackDetectionResult(
                        feedback_detected=True,
                        feedback_text="User gave thanks and asked a follow-up.",
                        feedback_score=5.0,
                        response_to_user="Thanks for your feedback!",
                        contains_followup_question=True,
                    ),
                ),
            ),
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = True
            mock_config_cls.return_value = mock_config
            mock_cache.get_latest_qa_entries.return_value = [
                SessionQAEntry(qa_id="last-qa-456", question="Q", context="", answer="A", time="t")
            ]

            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="thanks! What is the capital of France?",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert "Thanks for your feedback!" in result
        assert "Paris is the capital of France." in result
        mock_cache.update_qa_entry.assert_called_once()
        call_kw = mock_cache.update_qa_entry.call_args.kwargs
        assert call_kw["qa_id"] == "last-qa-456"
        mock_cache.create_qa_entry.assert_called_once()
        qa_kw = mock_cache.create_qa_entry.call_args.kwargs
        assert qa_kw["question"] == "thanks! What is the capital of France?"
        assert qa_kw["answer"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_auto_feedback_disabled_add_qa_no_feedback(
        self, sm, mock_cache
    ):
        """When caching True but auto_feedback False: add_qa called, no add_feedback."""
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
            patch(
                "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
                new_callable=AsyncMock,
                return_value=("Generated answer", "", None),
            ),
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = False
            mock_config_cls.return_value = mock_config

            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="Q?",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Generated answer"
        mock_cache.create_qa_entry.assert_called_once()
        mock_cache.update_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_feedback_detected_no_last_qa_id_adds_qa(
        self, sm, mock_cache
    ):
        """When feedback detected but no previous QA (empty session): add_qa called, no add_feedback."""
        mock_cache.get_latest_qa_entries.return_value = []
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
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
                        response_to_user="Thanks!",
                        contains_followup_question=False,
                    ),
                ),
            ),
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = True
            mock_config_cls.return_value = mock_config

            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="thanks!",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Generated answer"
        mock_cache.create_qa_entry.assert_called_once()
        mock_cache.update_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_feedback_persistence_failure_returns_response(
        self, sm, mock_cache
    ):
        """When feedback detected but add_feedback raises: still return response_to_user."""
        mock_cache.update_qa_entry = AsyncMock(side_effect=Exception("Cache write failed"))
        mock_cache.get_latest_qa_entries.return_value = [
            SessionQAEntry(qa_id="last-qa-789", question="Q", context="", answer="A", time="t")
        ]
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
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
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = True
            mock_config_cls.return_value = mock_config

            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="thanks!",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Thanks for your feedback!"
        mock_cache.create_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_feedback_only_empty_response_to_user_fallback(
        self, sm, mock_cache
    ):
        """When feedback only and response_to_user empty: return fallback thanks message."""
        mock_cache.get_latest_qa_entries.return_value = [
            SessionQAEntry(qa_id="last-qa-999", question="Q", context="", answer="A", time="t")
        ]
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
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
                        response_to_user="",
                        contains_followup_question=False,
                    ),
                ),
            ),
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = True
            mock_config_cls.return_value = mock_config

            result = await sm.generate_completion_with_session(
                session_id="s1",
                query="thanks!",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        assert result == "Thanks for your feedback."
        mock_cache.create_qa_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_completion_with_session_feedback_score_normalization(
        self, sm, mock_cache
    ):
        """Feedback score is normalized to int 1-5 when persisting."""
        mock_cache.get_latest_qa_entries.return_value = [
            SessionQAEntry(qa_id="last-qa-norm", question="Q", context="", answer="A", time="t")
        ]
        with (
            patch(
                "cognee.infrastructure.session.session_manager.session_user"
            ) as mock_session_user,
            patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
            patch(
                "cognee.infrastructure.session.session_manager.generate_session_completion_with_optional_summary",
                new_callable=AsyncMock,
                return_value=(
                    "Generated answer",
                    "",
                    FeedbackDetectionResult(
                        feedback_detected=True,
                        feedback_text="User gave 2.7 stars.",
                        feedback_score=2.7,
                        response_to_user="Thanks!",
                        contains_followup_question=False,
                    ),
                ),
            ),
        ):
            mock_user = MagicMock()
            mock_user.id = "u1"
            mock_session_user.get.return_value = mock_user
            mock_config = MagicMock()
            mock_config.caching = True
            mock_config.auto_feedback = True
            mock_config_cls.return_value = mock_config

            await sm.generate_completion_with_session(
                session_id="s1",
                query="2.5 stars",
                context="ctx",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
            )

        call_kw = mock_cache.update_qa_entry.call_args.kwargs
        assert call_kw["feedback_score"] == 3
        assert call_kw["feedback_text"] == "User gave 2.7 stars."
