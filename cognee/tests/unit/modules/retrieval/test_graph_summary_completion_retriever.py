import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


@pytest.fixture
def mock_edge():
    """Create a mock edge."""
    edge = MagicMock(spec=Edge)
    return edge


class TestGraphSummaryCompletionRetriever:
    @pytest.mark.asyncio
    async def test_init_defaults(self):
        """Test GraphSummaryCompletionRetriever initialization with defaults."""
        retriever = GraphSummaryCompletionRetriever()

        assert retriever.summarize_prompt_path == "summarize_search_results.txt"
        assert retriever.user_prompt_path == "graph_context_for_question.txt"
        assert retriever.system_prompt_path == "answer_simple_question.txt"
        assert retriever.top_k == 5
        assert retriever.save_interaction is False

    @pytest.mark.asyncio
    async def test_init_custom_params(self):
        """Test GraphSummaryCompletionRetriever initialization with custom parameters."""
        retriever = GraphSummaryCompletionRetriever(
            user_prompt_path="custom_user.txt",
            system_prompt_path="custom_system.txt",
            summarize_prompt_path="custom_summarize.txt",
            system_prompt="Custom system prompt",
            top_k=10,
            save_interaction=True,
            wide_search_top_k=200,
            triplet_distance_penalty=2.5,
        )

        assert retriever.summarize_prompt_path == "custom_summarize.txt"
        assert retriever.user_prompt_path == "custom_user.txt"
        assert retriever.system_prompt_path == "custom_system.txt"
        assert retriever.top_k == 10
        assert retriever.save_interaction is True

    @pytest.mark.asyncio
    async def test_resolve_edges_to_text_calls_super_and_summarizes(self, mock_edge):
        """Test resolve_edges_to_text calls super method and then summarizes."""
        retriever = GraphSummaryCompletionRetriever(
            summarize_prompt_path="custom_summarize.txt",
            system_prompt="Custom system prompt",
        )

        with (
            patch(
                "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text",
                new_callable=AsyncMock,
                return_value="Resolved edges text",
            ) as mock_super_resolve,
            patch(
                "cognee.modules.retrieval.graph_summary_completion_retriever.summarize_text",
                new_callable=AsyncMock,
                return_value="Summarized text",
            ) as mock_summarize,
        ):
            result = await retriever.resolve_edges_to_text([mock_edge])

            assert result == "Summarized text"
            mock_super_resolve.assert_awaited_once_with([mock_edge])
            mock_summarize.assert_awaited_once_with(
                "Resolved edges text",
                "custom_summarize.txt",
                "Custom system prompt",
            )

    @pytest.mark.asyncio
    async def test_resolve_edges_to_text_with_default_system_prompt(self, mock_edge):
        """Test resolve_edges_to_text uses None for system_prompt when not provided."""
        retriever = GraphSummaryCompletionRetriever()

        with (
            patch(
                "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text",
                new_callable=AsyncMock,
                return_value="Resolved edges text",
            ),
            patch(
                "cognee.modules.retrieval.graph_summary_completion_retriever.summarize_text",
                new_callable=AsyncMock,
                return_value="Summarized text",
            ) as mock_summarize,
        ):
            await retriever.resolve_edges_to_text([mock_edge])

            mock_summarize.assert_awaited_once_with(
                "Resolved edges text",
                "summarize_search_results.txt",
                None,
            )

    @pytest.mark.asyncio
    async def test_resolve_edges_to_text_with_empty_edges(self):
        """Test resolve_edges_to_text handles empty edges list."""
        retriever = GraphSummaryCompletionRetriever()

        with (
            patch(
                "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "cognee.modules.retrieval.graph_summary_completion_retriever.summarize_text",
                new_callable=AsyncMock,
                return_value="Empty summary",
            ) as mock_summarize,
        ):
            result = await retriever.resolve_edges_to_text([])

            assert result == "Empty summary"
            mock_summarize.assert_awaited_once_with(
                "",
                "summarize_search_results.txt",
                None,
            )

    @pytest.mark.asyncio
    async def test_resolve_edges_to_text_with_multiple_edges(self, mock_edge):
        """Test resolve_edges_to_text handles multiple edges."""
        retriever = GraphSummaryCompletionRetriever()

        mock_edge2 = MagicMock(spec=Edge)
        mock_edge3 = MagicMock(spec=Edge)

        with (
            patch(
                "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text",
                new_callable=AsyncMock,
                return_value="Multiple edges resolved text",
            ),
            patch(
                "cognee.modules.retrieval.graph_summary_completion_retriever.summarize_text",
                new_callable=AsyncMock,
                return_value="Multiple edges summarized",
            ) as mock_summarize,
        ):
            result = await retriever.resolve_edges_to_text([mock_edge, mock_edge2, mock_edge3])

            assert result == "Multiple edges summarized"
            mock_summarize.assert_awaited_once_with(
                "Multiple edges resolved text",
                "summarize_search_results.txt",
                None,
            )
