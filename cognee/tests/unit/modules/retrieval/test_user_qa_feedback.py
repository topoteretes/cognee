import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID, NAMESPACE_OID, uuid5

from cognee.modules.retrieval.user_qa_feedback import UserQAFeedback
from cognee.modules.retrieval.utils.models import UserFeedbackEvaluation, UserFeedbackSentiment
from cognee.modules.engine.models import NodeSet


@pytest.fixture
def mock_feedback_evaluation():
    """Create a mock feedback evaluation."""
    evaluation = MagicMock(spec=UserFeedbackEvaluation)
    evaluation.evaluation = MagicMock()
    evaluation.evaluation.value = "positive"
    evaluation.score = 4.5
    return evaluation


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine."""
    engine = AsyncMock()
    engine.get_last_user_interaction_ids = AsyncMock(return_value=[])
    engine.add_edges = AsyncMock()
    engine.apply_feedback_weight = AsyncMock()
    return engine


class TestUserQAFeedback:
    @pytest.mark.asyncio
    async def test_init_default(self):
        """Test UserQAFeedback initialization with default last_k."""
        retriever = UserQAFeedback()
        assert retriever.last_k == 1

    @pytest.mark.asyncio
    async def test_init_custom_last_k(self):
        """Test UserQAFeedback initialization with custom last_k."""
        retriever = UserQAFeedback(last_k=5)
        assert retriever.last_k == 5

    @pytest.mark.asyncio
    async def test_add_feedback_success_with_relationships(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback successfully creates feedback with relationships."""
        interaction_id_1 = str(UUID("550e8400-e29b-41d4-a716-446655440000"))
        interaction_id_2 = str(UUID("550e8400-e29b-41d4-a716-446655440001"))
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(
            return_value=[interaction_id_1, interaction_id_2]
        )

        feedback_text = "This answer was helpful"

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ) as mock_add_data,
            patch(
                "cognee.modules.retrieval.user_qa_feedback.index_graph_edges",
                new_callable=AsyncMock,
            ) as mock_index_edges,
        ):
            retriever = UserQAFeedback(last_k=2)
            result = await retriever.add_feedback(feedback_text)

            assert result == [feedback_text]
            mock_add_data.assert_awaited_once()
            mock_graph_engine.add_edges.assert_awaited_once()
            mock_index_edges.assert_awaited_once()
            mock_graph_engine.apply_feedback_weight.assert_awaited_once()

            # Verify add_edges was called with correct relationships
            call_args = mock_graph_engine.add_edges.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0][0] == uuid5(NAMESPACE_OID, name=feedback_text)
            assert call_args[0][1] == UUID(interaction_id_1)
            assert call_args[0][2] == "gives_feedback_to"
            assert call_args[0][3]["relationship_name"] == "gives_feedback_to"
            assert call_args[0][3]["ontology_valid"] is False

            # Verify apply_feedback_weight was called with correct node IDs
            weight_call_args = mock_graph_engine.apply_feedback_weight.call_args[1]["node_ids"]
            assert len(weight_call_args) == 2
            assert interaction_id_1 in weight_call_args
            assert interaction_id_2 in weight_call_args

    @pytest.mark.asyncio
    async def test_add_feedback_success_no_relationships(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback successfully creates feedback without relationships."""
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(return_value=[])

        feedback_text = "This answer was helpful"

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ) as mock_add_data,
            patch(
                "cognee.modules.retrieval.user_qa_feedback.index_graph_edges",
                new_callable=AsyncMock,
            ) as mock_index_edges,
        ):
            retriever = UserQAFeedback(last_k=1)
            result = await retriever.add_feedback(feedback_text)

            assert result == [feedback_text]
            mock_add_data.assert_awaited_once()
            # Should not call add_edges or index_graph_edges when no relationships
            mock_graph_engine.add_edges.assert_not_awaited()
            mock_index_edges.assert_not_awaited()
            mock_graph_engine.apply_feedback_weight.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_feedback_creates_correct_feedback_node(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback creates CogneeUserFeedback with correct attributes."""
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(return_value=[])

        feedback_text = "This was a negative experience"
        mock_feedback_evaluation.evaluation.value = "negative"
        mock_feedback_evaluation.score = -3.0

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ) as mock_add_data,
        ):
            retriever = UserQAFeedback()
            await retriever.add_feedback(feedback_text)

            # Verify add_data_points was called with correct CogneeUserFeedback
            call_args = mock_add_data.call_args[1]["data_points"]
            assert len(call_args) == 1
            feedback_node = call_args[0]
            assert feedback_node.id == uuid5(NAMESPACE_OID, name=feedback_text)
            assert feedback_node.feedback == feedback_text
            assert feedback_node.sentiment == "negative"
            assert feedback_node.score == -3.0
            assert isinstance(feedback_node.belongs_to_set, NodeSet)
            assert feedback_node.belongs_to_set.name == "UserQAFeedbacks"

    @pytest.mark.asyncio
    async def test_add_feedback_calls_llm_with_correct_prompt(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback calls LLM with correct sentiment analysis prompt."""
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(return_value=[])

        feedback_text = "Great answer!"

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ) as mock_llm,
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ),
        ):
            retriever = UserQAFeedback()
            await retriever.add_feedback(feedback_text)

            mock_llm.assert_awaited_once()
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["text_input"] == feedback_text
            assert "sentiment analysis assistant" in call_kwargs["system_prompt"]
            assert call_kwargs["response_model"] == UserFeedbackEvaluation

    @pytest.mark.asyncio
    async def test_add_feedback_uses_last_k_parameter(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback uses last_k parameter when getting interaction IDs."""
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(return_value=[])

        feedback_text = "Test feedback"

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ),
        ):
            retriever = UserQAFeedback(last_k=5)
            await retriever.add_feedback(feedback_text)

            mock_graph_engine.get_last_user_interaction_ids.assert_awaited_once_with(limit=5)

    @pytest.mark.asyncio
    async def test_add_feedback_with_single_interaction(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback with single interaction ID."""
        interaction_id = str(UUID("550e8400-e29b-41d4-a716-446655440000"))
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(return_value=[interaction_id])

        feedback_text = "Test feedback"

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.index_graph_edges",
                new_callable=AsyncMock,
            ),
        ):
            retriever = UserQAFeedback()
            result = await retriever.add_feedback(feedback_text)

            assert result == [feedback_text]
            # Should create relationship for the interaction
            call_args = mock_graph_engine.add_edges.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0][1] == UUID(interaction_id)

    @pytest.mark.asyncio
    async def test_add_feedback_applies_weight_correctly(
        self, mock_feedback_evaluation, mock_graph_engine
    ):
        """Test add_feedback applies feedback weight with correct score."""
        interaction_id = str(UUID("550e8400-e29b-41d4-a716-446655440000"))
        mock_graph_engine.get_last_user_interaction_ids = AsyncMock(return_value=[interaction_id])
        mock_feedback_evaluation.score = 4.5

        feedback_text = "Positive feedback"

        with (
            patch(
                "cognee.modules.retrieval.user_qa_feedback.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=mock_feedback_evaluation,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.get_graph_engine",
                return_value=mock_graph_engine,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.add_data_points",
                new_callable=AsyncMock,
            ),
            patch(
                "cognee.modules.retrieval.user_qa_feedback.index_graph_edges",
                new_callable=AsyncMock,
            ),
        ):
            retriever = UserQAFeedback()
            await retriever.add_feedback(feedback_text)

            mock_graph_engine.apply_feedback_weight.assert_awaited_once_with(
                node_ids=[interaction_id], weight=4.5
            )
