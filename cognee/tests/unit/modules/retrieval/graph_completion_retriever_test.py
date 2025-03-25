from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.exceptions import EntityNotFoundError
from cognee.tasks.completion.exceptions import NoRelevantDataFound


class TestGraphCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return GraphCompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search")
    async def test_get_triplets_success(self, mock_brute_force_triplet_search, mock_retriever):
        mock_brute_force_triplet_search.return_value = [
            AsyncMock(
                node1=AsyncMock(attributes={"text": "Node A"}),
                attributes={"relationship_type": "connects"},
                node2=AsyncMock(attributes={"text": "Node B"}),
            )
        ]

        result = await mock_retriever.get_triplets("test query")

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].attributes["relationship_type"] == "connects"
        mock_brute_force_triplet_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search")
    async def test_get_triplets_no_results(self, mock_brute_force_triplet_search, mock_retriever):
        mock_brute_force_triplet_search.return_value = []

        with pytest.raises(NoRelevantDataFound):
            await mock_retriever.get_triplets("test query")

    @pytest.mark.asyncio
    async def test_resolve_edges_to_text(self, mock_retriever):
        node_a = AsyncMock(id="node_a_id", attributes={"text": "Node A text content"})
        node_b = AsyncMock(id="node_b_id", attributes={"text": "Node B text content"})
        node_c = AsyncMock(id="node_c_id", attributes={"name": "Node C"})

        triplets = [
            AsyncMock(
                node1=node_a,
                attributes={"relationship_type": "connects"},
                node2=node_b,
            ),
            AsyncMock(
                node1=node_a,
                attributes={"relationship_type": "links"},
                node2=node_c,
            ),
        ]

        with patch.object(mock_retriever, "_get_title", return_value="Test Title"):
            result = await mock_retriever.resolve_edges_to_text(triplets)

            assert "Nodes:" in result
            assert "Connections:" in result

            assert "Node: Test Title" in result
            assert "__node_content_start__" in result
            assert "Node A text content" in result
            assert "__node_content_end__" in result
            assert "Node: Node C" in result

            assert "Test Title --[connects]--> Test Title" in result
            assert "Test Title --[links]--> Node C" in result

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_triplets",
        new_callable=AsyncMock,
    )
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text",
        new_callable=AsyncMock,
    )
    async def test_get_context(self, mock_resolve_edges_to_text, mock_get_triplets, mock_retriever):
        """Test get_context calls get_triplets and resolve_edges_to_text."""
        mock_get_triplets.return_value = ["mock_triplet"]
        mock_resolve_edges_to_text.return_value = "Mock Context"

        result = await mock_retriever.get_context("test query")

        assert result == "Mock Context"
        mock_get_triplets.assert_called_once_with("test query")
        mock_resolve_edges_to_text.assert_called_once_with(["mock_triplet"])

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_context"
    )
    @patch("cognee.modules.retrieval.graph_completion_retriever.generate_completion")
    async def test_get_completion_without_context(
        self, mock_generate_completion, mock_get_context, mock_retriever
    ):
        """Test get_completion when no context is provided (calls get_context)."""
        mock_get_context.return_value = "Mock Context"
        mock_generate_completion.return_value = "Generated Completion"

        result = await mock_retriever.get_completion("test query")

        assert result == ["Generated Completion"]
        mock_get_context.assert_called_once_with("test query")
        mock_generate_completion.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_context"
    )
    @patch("cognee.modules.retrieval.graph_completion_retriever.generate_completion")
    async def test_get_completion_with_context(
        self, mock_generate_completion, mock_get_context, mock_retriever
    ):
        """Test get_completion when context is provided (does not call get_context)."""
        mock_generate_completion.return_value = "Generated Completion"

        result = await mock_retriever.get_completion("test query", context="Provided Context")

        assert result == ["Generated Completion"]
        mock_get_context.assert_not_called()
        mock_generate_completion.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine")
    @patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_default_user")
    async def test_get_completion_with_empty_graph(
        self,
        mock_get_default_user,
        mock_get_graph_engine,
        mock_get_llm_client,
        mock_retriever,
    ):
        query = "test query with empty graph"

        mock_graph_engine = MagicMock()
        mock_graph_engine.get_graph_data = AsyncMock()
        mock_graph_engine.get_graph_data.return_value = ([], [])
        mock_get_graph_engine.return_value = mock_graph_engine

        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = (
            "Generated graph completion response"
        )
        mock_get_llm_client.return_value = mock_llm_client

        with pytest.raises(EntityNotFoundError):
            await mock_retriever.get_completion(query)

        mock_graph_engine.get_graph_data.assert_called_once()

    def test_top_n_words(self, mock_retriever):
        """Test extraction of top frequent words from text."""
        text = "The quick brown fox jumps over the lazy dog. The fox is quick."

        result = mock_retriever._top_n_words(text)
        assert len(result.split(", ")) <= 3
        assert "fox" in result
        assert "quick" in result

        result = mock_retriever._top_n_words(text, top_n=2)
        assert len(result.split(", ")) <= 2

        result = mock_retriever._top_n_words(text, separator=" | ")
        assert " | " in result

        result = mock_retriever._top_n_words(text, stop_words={"fox", "quick"})
        assert "fox" not in result
        assert "quick" not in result

    def test_get_title(self, mock_retriever):
        """Test title generation from text."""
        text = "This is a long paragraph about various topics that should generate a title. The main topics are AI, programming and data science."

        title = mock_retriever._get_title(text)
        assert "..." in title
        assert "[" in title and "]" in title

        title = mock_retriever._get_title(text, first_n_words=3)
        first_part = title.split("...")[0].strip()
        assert len(first_part.split()) == 3

        title = mock_retriever._get_title(text, top_n_words=2)
        top_part = title.split("[")[1].split("]")[0]
        assert len(top_part.split(", ")) <= 2

    def test_get_nodes(self, mock_retriever):
        """Test node processing and deduplication."""
        node_with_text = AsyncMock(id="text_node", attributes={"text": "This is a text node"})
        node_with_name = AsyncMock(id="name_node", attributes={"name": "Named Node"})
        node_without_attrs = AsyncMock(id="empty_node", attributes={})

        edges = [
            AsyncMock(
                node1=node_with_text, node2=node_with_name, attributes={"relationship_type": "rel1"}
            ),
            AsyncMock(
                node1=node_with_text,
                node2=node_without_attrs,
                attributes={"relationship_type": "rel2"},
            ),
            AsyncMock(
                node1=node_with_name,
                node2=node_without_attrs,
                attributes={"relationship_type": "rel3"},
            ),
        ]

        with patch.object(mock_retriever, "_get_title", return_value="Generated Title"):
            nodes = mock_retriever._get_nodes(edges)

            assert len(nodes) == 3

            for node_id, info in nodes.items():
                assert "node" in info
                assert "name" in info
                assert "content" in info

            text_node_info = nodes[node_with_text.id]
            assert text_node_info["name"] == "Generated Title"
            assert text_node_info["content"] == "This is a text node"

            name_node_info = nodes[node_with_name.id]
            assert name_node_info["name"] == "Named Node"
            assert name_node_info["content"] == "Named Node"

            empty_node_info = nodes[node_without_attrs.id]
            assert empty_node_info["name"] == "Unnamed Node"
