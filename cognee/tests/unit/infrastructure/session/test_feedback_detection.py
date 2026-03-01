import pytest
from unittest.mock import AsyncMock, patch

from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult
from cognee.infrastructure.session.feedback_detection import detect_feedback


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
    async def test_detect_feedback_llm_returns_wrong_type_returns_not_detected(self):
        """When LLM returns non-FeedbackDetectionResult, returns feedback_detected=False."""
        with (
            patch(
                "cognee.infrastructure.session.feedback_detection.read_query_prompt",
                return_value="System prompt",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value={"feedback_detected": True},
            ),
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
