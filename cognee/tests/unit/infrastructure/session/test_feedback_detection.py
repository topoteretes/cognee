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
    async def test_detect_feedback_empty_string_returns_not_detected(self):
        """Empty string returns feedback_detected=False without calling LLM."""
        result = await detect_feedback("")
        assert result.feedback_detected is False
        assert result.feedback_text is None
        assert result.feedback_score is None

    @pytest.mark.asyncio
    async def test_detect_feedback_whitespace_returns_not_detected(self):
        """Whitespace-only string returns feedback_detected=False without calling LLM."""
        result = await detect_feedback("   \n\t  ")
        assert result.feedback_detected is False

    @pytest.mark.asyncio
    async def test_detect_feedback_llm_returns_detected(self):
        """When LLM returns feedback_detected=True, that result is returned."""
        expected = FeedbackDetectionResult(
            feedback_detected=True,
            feedback_text="User said thanks.",
            feedback_score=5.0,
            response_to_user="Thanks for your feedback!",
            contains_followup_question=False,
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
            result = await detect_feedback("thanks, that was helpful!")

        assert result.feedback_detected is True
        assert result.feedback_text == "User said thanks."
        assert result.feedback_score == 5.0
        assert result.response_to_user == "Thanks for your feedback!"
        mock_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_detect_feedback_llm_returns_not_detected(self):
        """When LLM returns feedback_detected=False, that result is returned."""
        expected = FeedbackDetectionResult(feedback_detected=False)
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
            result = await detect_feedback("What is the capital of France?")

        assert result.feedback_detected is False

    @pytest.mark.asyncio
    async def test_detect_feedback_llm_raises_returns_not_detected(self):
        """When LLM raises, returns feedback_detected=False so main flow is not blocked."""
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

        assert result.feedback_detected is False

    @pytest.mark.asyncio
    async def test_detect_feedback_prompt_missing_returns_not_detected(self):
        """When read_query_prompt returns None/empty, returns feedback_detected=False."""
        with patch(
            "cognee.infrastructure.session.feedback_detection.read_query_prompt",
            return_value=None,
        ):
            result = await detect_feedback("thanks!")

        assert result.feedback_detected is False

    @pytest.mark.asyncio
    async def test_detect_feedback_contains_followup_question_true(self):
        """When LLM returns contains_followup_question=True, it is preserved."""
        expected = FeedbackDetectionResult(
            feedback_detected=True,
            feedback_text="Thanks and follow-up.",
            feedback_score=5.0,
            response_to_user="Thanks!",
            contains_followup_question=True,
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

        assert result.contains_followup_question is True

    def test_prompt_extracts_candidate_updates_without_feedback(self):
        """Prompt tells the analyzer to extract durable guidance from non-feedback messages."""
        prompt_path = (
            Path(__file__).parents[4]
            / "infrastructure"
            / "llm"
            / "prompts"
            / "feedback_detection_system.txt"
        )
        prompt = prompt_path.read_text()

        assert "candidate_context_updates are independent of previous_answer_feedback" in prompt
        assert "answer style, answer length, or answer format preference" in prompt
        assert "For now, answer with 2 informative bullet points." in prompt
        assert "I now prefer 4 concise bullet points" in prompt


class TestDetectFeedbackServedContext:
    """Tests for the served_context wiring into the single feedback LLM call."""

    @pytest.mark.asyncio
    async def test_served_context_list_injected_into_text_input(self):
        """A list of served entries is rendered as 'id: content' and appended to text_input."""
        expected = FeedbackDetectionResult(
            feedback_detected=True,
            feedback_text="User corrected the answer.",
            feedback_score=2.0,
            response_to_user="Thanks for the correction!",
            contains_followup_question=False,
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

        assert result.feedback_detected is True
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
        expected = FeedbackDetectionResult(feedback_detected=False)
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
        expected = FeedbackDetectionResult(feedback_detected=False)
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
        expected = FeedbackDetectionResult(feedback_detected=False)
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
        """FeedbackDetectionResult(feedback_detected=False) still validates with empty defaults."""
        result = FeedbackDetectionResult(feedback_detected=False)
        assert result.feedback_detected is False
        assert result.followup_question is None
        assert result.served_context_ratings == []
        assert result.candidate_context_updates == []

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
        result = FeedbackDetectionResult(feedback_detected=True, served_context_ratings=ratings)
        assert len(result.served_context_ratings) == 3
        assert [r.entry_id for r in result.served_context_ratings] == ["id-0", "id-1", "id-2"]

    def test_candidate_context_updates_truncate_to_three(self):
        """5 candidate context updates truncate to the first 3."""
        candidates = [
            CandidateContextUpdate(section="rules", content=f"rule {i}", confidence=0.9)
            for i in range(5)
        ]
        result = FeedbackDetectionResult(
            feedback_detected=True, candidate_context_updates=candidates
        )
        assert len(result.candidate_context_updates) == 3
        assert [c.content for c in result.candidate_context_updates] == [
            "rule 0",
            "rule 1",
            "rule 2",
        ]
