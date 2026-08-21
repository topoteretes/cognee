from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.session.feedback_detection import detect_feedback
from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult
from cognee.infrastructure.session.session_context_models import (
    CandidateContextUpdate,
    ServedContextRating,
)


class TestDetectFeedback:
    """Tests for detect_feedback."""

    @pytest.mark.asyncio
    async def test_detect_feedback_empty_string_returns_empty_analysis(self):
        """Empty string returns an empty analysis without calling LLM."""
        result = await detect_feedback("")
        assert result.query_to_answer is None
        assert result.response_to_user is None

    @pytest.mark.asyncio
    async def test_detect_feedback_whitespace_returns_empty_analysis(self):
        """Whitespace-only string returns an empty analysis without calling LLM."""
        result = await detect_feedback("   \n\t  ")
        assert result.query_to_answer is None

    @pytest.mark.asyncio
    async def test_detect_feedback_llm_returns_query_to_answer(self):
        """When LLM returns query_to_answer, that result is returned."""
        expected = FeedbackDetectionResult(
            query_to_answer="What is the capital of France?",
        )
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ) as mock_llm,
        ):
            result = await detect_feedback("What is the capital of France?")

        assert result.query_to_answer == "What is the capital of France?"
        mock_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_detect_feedback_llm_returns_no_query_to_answer(self):
        """When LLM returns no query_to_answer, that result is returned."""
        expected = FeedbackDetectionResult(response_to_user="Got it.")
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ),
        ):
            result = await detect_feedback("that was wrong")

        assert result.query_to_answer is None
        assert result.response_to_user == "Got it."

    @pytest.mark.asyncio
    async def test_detect_feedback_llm_raises_returns_empty_analysis(self):
        """When LLM raises, returns empty analysis so main flow is not blocked."""
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                side_effect=Exception("LLM timeout"),
            ),
        ):
            result = await detect_feedback("thanks!")

        assert result.query_to_answer is None

    @pytest.mark.asyncio
    async def test_detect_feedback_prompt_missing_returns_empty_analysis(self):
        """When read_query_prompt returns None/empty, returns empty analysis."""
        with patch(
            "cognee.infrastructure.session.feedback_detection.read_query_prompt",
            return_value=None,
        ):
            result = await detect_feedback("thanks!")

        assert result.query_to_answer is None

    @pytest.mark.asyncio
    async def test_detect_feedback_query_to_answer_strips_feedback_prefix(self):
        """LLM-provided query_to_answer is preserved for feedback plus follow-up turns."""
        expected = FeedbackDetectionResult(
            query_to_answer="What is the capital of France?",
        )
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ),
        ):
            result = await detect_feedback("thanks! What is the capital of France?")

        assert result.query_to_answer == "What is the capital of France?"

    def test_prompt_contains_session_turn_analysis_anchors(self):
        """Prompt keeps the stable sections and output fields used by the analyzer."""
        prompt_path = (
            Path(__file__).parents[4]
            / "infrastructure"
            / "llm"
            / "prompts"
            / "feedback_detection_system.txt"
        )
        prompt = prompt_path.read_text()

        assert "## 1. Query to answer" in prompt
        assert "## 2. Candidate session-context updates" in prompt
        assert "## 3. Served context ratings" in prompt
        assert "query_to_answer" in prompt

    def test_candidate_context_updates_parse_section_specific_models(self):
        result = FeedbackDetectionResult(
            candidate_context_updates=[
                {
                    "section": " Rules ",
                    "content": "Use PostgreSQL for database examples.",
                    "confidence": 0.9,
                },
                {
                    "section": " LESSONS_LEARNED ",
                    "content": "The previous Docker build failed due to memory limits.",
                    "confidence": 0.85,
                },
            ]
        )

        # Both items parse as the flat base class — no discriminated union dispatch.
        assert isinstance(result.candidate_context_updates[0], CandidateContextUpdate)
        assert isinstance(result.candidate_context_updates[1], CandidateContextUpdate)
        assert result.candidate_context_updates[0].section == "rules"
        assert result.candidate_context_updates[1].section == "lessons_learned"


class TestDetectFeedbackServedContext:
    """Tests for the served_context wiring into the single feedback LLM call."""

    @pytest.mark.asyncio
    async def test_served_context_list_injected_into_text_input(self):
        """A list of served entries is rendered as 'id: content' and appended to text_input."""
        expected = FeedbackDetectionResult(
            response_to_user="Thanks for the correction!",
            served_context_ratings=[ServedContextRating(entry_id="ctx-1", rating="harmful")],
            candidate_context_updates=[
                CandidateContextUpdate(
                    section="rules", content="Always cite sources.", confidence=0.9
                )
            ],
        )
        served = [
            {"id": "ctx-1", "content": "Prefer concise answers."},
            {"id": "ctx-2", "content": "Use metric units."},
        ]
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ) as mock_llm,
        ):
            result = await detect_feedback("that was wrong", served_context=served)

        assert result.query_to_answer is None
        assert [r.entry_id for r in result.served_context_ratings] == ["ctx-1"]
        assert result.candidate_context_updates[0].section == "rules"
        text_input = mock_llm.await_args.kwargs["text_input"]
        assert "that was wrong" in text_input
        assert "SESSION CONTEXT ENTRIES SERVED TO THE PREVIOUS ANSWER (id: content):" in text_input
        assert "ctx-1: Prefer concise answers." in text_input
        assert "ctx-2: Use metric units." in text_input

    @pytest.mark.asyncio
    async def test_served_context_string_injected_into_text_input(self):
        """A pre-rendered string served_context is appended verbatim."""
        expected = FeedbackDetectionResult()
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ) as mock_llm,
        ):
            await detect_feedback("nice", served_context="ctx-9: Keep it short.")

        text_input = mock_llm.await_args.kwargs["text_input"]
        assert "ctx-9: Keep it short." in text_input

    @pytest.mark.asyncio
    async def test_none_served_context_leaves_text_input_unchanged(self):
        """With served_context=None, text_input is just the stripped user message."""
        expected = FeedbackDetectionResult()
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ) as mock_llm,
        ):
            await detect_feedback("  what is X?  ")

        text_input = mock_llm.await_args.kwargs["text_input"]
        assert text_input == "CURRENT USER MESSAGE:\nwhat is X?"
        assert "SESSION CONTEXT ENTRIES" not in text_input

    @pytest.mark.asyncio
    async def test_empty_served_context_list_leaves_text_input_unchanged(self):
        """An empty served_context list renders nothing and is not appended."""
        expected = FeedbackDetectionResult()
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=expected,
            ) as mock_llm,
        ):
            await detect_feedback("hello", served_context=[])

        text_input = mock_llm.await_args.kwargs["text_input"]
        assert text_input == "CURRENT USER MESSAGE:\nhello"
        assert "SESSION CONTEXT ENTRIES" not in text_input


class TestFeedbackDetectionResultModel:
    """Tests for the FeedbackDetectionResult pydantic model."""

    def test_empty_construction_still_validates(self):
        """FeedbackDetectionResult still validates with empty defaults."""
        result = FeedbackDetectionResult()
        assert result.query_to_answer is None
        assert result.served_context_ratings == []
        assert result.candidate_context_updates == []

    def test_optional_text_fields_normalize_blank_to_none(self):
        """Blank optional text fields normalize to None."""
        result = FeedbackDetectionResult(response_to_user="  ", query_to_answer="\n")
        assert result.response_to_user is None
        assert result.query_to_answer is None

    def test_defaults_are_empty_lists(self):
        """New list fields default to independent empty lists."""
        a = FeedbackDetectionResult()
        b = FeedbackDetectionResult()
        assert a.served_context_ratings == []
        assert a.candidate_context_updates == []
        a.served_context_ratings.append(ServedContextRating(entry_id="x", rating="helpful"))
        assert b.served_context_ratings == []

    def test_served_context_ratings_truncate_to_three(self):
        """5 served-context ratings truncate to the first 3."""
        ratings = [ServedContextRating(entry_id=f"id-{i}", rating="helpful") for i in range(5)]
        result = FeedbackDetectionResult(served_context_ratings=ratings)
        assert len(result.served_context_ratings) == 3
        assert [r.entry_id for r in result.served_context_ratings] == ["id-0", "id-1", "id-2"]

    def test_candidate_context_updates_truncate_to_three(self):
        """5 candidate context updates truncate to the first 3."""
        candidates = [
            CandidateContextUpdate(section="rules", content=f"rule {i}", confidence=0.9)
            for i in range(5)
        ]
        result = FeedbackDetectionResult(candidate_context_updates=candidates)
        assert len(result.candidate_context_updates) == 3
        assert [c.content for c in result.candidate_context_updates] == [
            "rule 0",
            "rule 1",
            "rule 2",
        ]
