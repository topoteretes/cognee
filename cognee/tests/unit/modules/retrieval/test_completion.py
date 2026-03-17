import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Type


class TestGenerateCompletion:
    @pytest.mark.asyncio
    async def test_generate_completion_with_system_prompt(self):
        """Test generate_completion with provided system_prompt."""
        mock_llm_response = "Generated answer"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import generate_completion

            result = await generate_completion(
                query="What is AI?",
                context="AI is artificial intelligence",
                user_prompt_path="user_prompt.txt",
                system_prompt_path="system_prompt.txt",
                system_prompt="Custom system prompt",
            )

            assert result == mock_llm_response
            mock_llm.assert_awaited_once_with(
                text_input="User prompt text",
                system_prompt="Custom system prompt",
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_generate_completion_without_system_prompt(self):
        """Test generate_completion reads system_prompt from file when not provided."""
        mock_llm_response = "Generated answer"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import generate_completion

            result = await generate_completion(
                query="What is AI?",
                context="AI is artificial intelligence",
                user_prompt_path="user_prompt.txt",
                system_prompt_path="system_prompt.txt",
            )

            assert result == mock_llm_response
            mock_llm.assert_awaited_once_with(
                text_input="User prompt text",
                system_prompt="System prompt from file",
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_generate_completion_with_conversation_history(self):
        """Test generate_completion includes conversation_history in system_prompt."""
        mock_llm_response = "Generated answer"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import generate_completion

            result = await generate_completion(
                query="What is AI?",
                context="AI is artificial intelligence",
                user_prompt_path="user_prompt.txt",
                system_prompt_path="system_prompt.txt",
                conversation_history="Previous conversation:\nQ: What is ML?\nA: ML is machine learning",
            )

            assert result == mock_llm_response
            expected_system_prompt = (
                "Previous conversation:\nQ: What is ML?\nA: ML is machine learning"
                + "\nTASK:"
                + "System prompt from file"
            )
            mock_llm.assert_awaited_once_with(
                text_input="User prompt text",
                system_prompt=expected_system_prompt,
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_generate_completion_with_conversation_history_and_custom_system_prompt(self):
        """Test generate_completion includes conversation_history with custom system_prompt."""
        mock_llm_response = "Generated answer"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import generate_completion

            result = await generate_completion(
                query="What is AI?",
                context="AI is artificial intelligence",
                user_prompt_path="user_prompt.txt",
                system_prompt_path="system_prompt.txt",
                system_prompt="Custom system prompt",
                conversation_history="Previous conversation:\nQ: What is ML?\nA: ML is machine learning",
            )

            assert result == mock_llm_response
            expected_system_prompt = (
                "Previous conversation:\nQ: What is ML?\nA: ML is machine learning"
                + "\nTASK:"
                + "Custom system prompt"
            )
            mock_llm.assert_awaited_once_with(
                text_input="User prompt text",
                system_prompt=expected_system_prompt,
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_generate_completion_with_response_model(self):
        """Test generate_completion with custom response_model."""
        mock_response_model = MagicMock()
        mock_llm_response = {"answer": "Generated answer"}

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import generate_completion

            result = await generate_completion(
                query="What is AI?",
                context="AI is artificial intelligence",
                user_prompt_path="user_prompt.txt",
                system_prompt_path="system_prompt.txt",
                response_model=mock_response_model,
            )

            assert result == mock_llm_response
            mock_llm.assert_awaited_once_with(
                text_input="User prompt text",
                system_prompt="System prompt from file",
                response_model=mock_response_model,
            )

    @pytest.mark.asyncio
    async def test_generate_completion_render_prompt_args(self):
        """Test generate_completion passes correct args to render_prompt."""
        mock_llm_response = "Generated answer"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ) as mock_render,
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
        ):
            from cognee.modules.retrieval.utils.completion import generate_completion

            await generate_completion(
                query="What is AI?",
                context="AI is artificial intelligence",
                user_prompt_path="user_prompt.txt",
                system_prompt_path="system_prompt.txt",
            )

            mock_render.assert_called_once_with(
                "user_prompt.txt",
                {"question": "What is AI?", "context": "AI is artificial intelligence"},
            )


class TestSummarizeText:
    @pytest.mark.asyncio
    async def test_summarize_text_with_system_prompt(self):
        """Test summarize_text with provided system_prompt."""
        mock_llm_response = "Summary text"

        with patch(
            "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ) as mock_llm:
            from cognee.modules.retrieval.utils.completion import summarize_text

            result = await summarize_text(
                text="Long text to summarize",
                system_prompt_path="summarize_search_results.txt",
                system_prompt="Custom summary prompt",
            )

            assert result == mock_llm_response
            mock_llm.assert_awaited_once_with(
                text_input="Long text to summarize",
                system_prompt="Custom summary prompt",
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_summarize_text_without_system_prompt(self):
        """Test summarize_text reads system_prompt from file when not provided."""
        mock_llm_response = "Summary text"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import summarize_text

            result = await summarize_text(
                text="Long text to summarize",
                system_prompt_path="summarize_search_results.txt",
            )

            assert result == mock_llm_response
            mock_llm.assert_awaited_once_with(
                text_input="Long text to summarize",
                system_prompt="System prompt from file",
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_summarize_text_default_prompt_path(self):
        """Test summarize_text uses default prompt path when not provided."""
        mock_llm_response = "Summary text"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="Default system prompt",
            ) as mock_read,
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import summarize_text

            result = await summarize_text(text="Long text to summarize")

            assert result == mock_llm_response
            mock_read.assert_called_once_with("summarize_search_results.txt")
            mock_llm.assert_awaited_once_with(
                text_input="Long text to summarize",
                system_prompt="Default system prompt",
                response_model=str,
            )

    @pytest.mark.asyncio
    async def test_summarize_text_custom_prompt_path(self):
        """Test summarize_text uses custom prompt path when provided."""
        mock_llm_response = "Summary text"

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="Custom system prompt",
            ) as mock_read,
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import summarize_text

            result = await summarize_text(
                text="Long text to summarize",
                system_prompt_path="custom_prompt.txt",
            )

            assert result == mock_llm_response
            mock_read.assert_called_once_with("custom_prompt.txt")
            mock_llm.assert_awaited_once_with(
                text_input="Long text to summarize",
                system_prompt="Custom system prompt",
                response_model=str,
            )


class TestGenerateSessionCompletionWithOptionalSummary:
    """Tests for generate_session_completion_with_optional_summary (session + optional feedback)."""

    @pytest.mark.asyncio
    async def test_run_feedback_detection_false_returns_none_feedback(self):
        """When run_feedback_detection=False, returns (completion, context_to_store, None)."""
        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value="Generated answer",
            ) as mock_llm,
        ):
            from cognee.modules.retrieval.utils.completion import (
                generate_session_completion_with_optional_summary,
            )

            (
                completion,
                context_to_store,
                feedback_result,
            ) = await generate_session_completion_with_optional_summary(
                query="Q?",
                context="ctx",
                conversation_history="",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
                summarize_context=False,
                run_feedback_detection=False,
            )

        assert completion == "Generated answer"
        assert context_to_store == ""
        assert feedback_result is None
        assert mock_llm.await_count == 1

    @pytest.mark.asyncio
    async def test_run_feedback_detection_true_runs_detect_feedback_in_parallel(self):
        """When run_feedback_detection=True, runs completion and detect_feedback in parallel."""
        from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value="Generated answer",
            ) as mock_llm,
            patch(
                "cognee.infrastructure.session.feedback_detection.detect_feedback",
                new_callable=AsyncMock,
                return_value=FeedbackDetectionResult(
                    feedback_detected=True,
                    feedback_text="User said thanks.",
                    feedback_score=5.0,
                    response_to_user="Thanks!",
                    contains_followup_question=False,
                ),
            ) as mock_detect,
        ):
            from cognee.modules.retrieval.utils.completion import (
                generate_session_completion_with_optional_summary,
            )

            (
                completion,
                context_to_store,
                feedback_result,
            ) = await generate_session_completion_with_optional_summary(
                query="thanks!",
                context="ctx",
                conversation_history="",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
                summarize_context=False,
                run_feedback_detection=True,
            )

        assert completion == "Generated answer"
        assert context_to_store == ""
        assert feedback_result is not None
        assert feedback_result.feedback_detected is True
        assert feedback_result.feedback_text == "User said thanks."
        mock_llm.assert_awaited_once()
        mock_detect.assert_awaited_once_with("thanks!")

    @pytest.mark.asyncio
    async def test_summarize_context_true_run_feedback_detection_true_returns_three_tuple(self):
        """With summarize_context and run_feedback_detection True, returns (completion, summary, feedback_result)."""
        from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult

        with (
            patch(
                "cognee.modules.retrieval.utils.completion.render_prompt",
                return_value="User prompt text",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.read_query_prompt",
                return_value="System prompt from file",
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.summarize_text",
                new_callable=AsyncMock,
                return_value="Summarized context",
            ) as mock_summarize,
            patch(
                "cognee.modules.retrieval.utils.completion.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value="Generated answer",
            ),
            patch(
                "cognee.infrastructure.session.feedback_detection.detect_feedback",
                new_callable=AsyncMock,
                return_value=FeedbackDetectionResult(feedback_detected=False),
            ) as mock_detect,
        ):
            from cognee.modules.retrieval.utils.completion import (
                generate_session_completion_with_optional_summary,
            )

            (
                completion,
                context_to_store,
                feedback_result,
            ) = await generate_session_completion_with_optional_summary(
                query="Q?",
                context="long context",
                conversation_history="",
                user_prompt_path="user.txt",
                system_prompt_path="sys.txt",
                summarize_context=True,
                run_feedback_detection=True,
            )

        assert completion == "Generated answer"
        assert context_to_store == "Summarized context"
        assert feedback_result is not None
        assert feedback_result.feedback_detected is False
        mock_summarize.assert_awaited_once()
        mock_detect.assert_awaited_once_with("Q?")
