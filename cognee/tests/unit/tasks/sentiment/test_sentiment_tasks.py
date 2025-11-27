from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from cognee.tasks.sentiment.extract_recent_interactions import extract_recent_interactions
from cognee.tasks.sentiment.classify_interaction_sentiment import classify_interaction_sentiment
from cognee.tasks.sentiment.link_sentiment_to_interactions import link_sentiment_to_interactions
from cognee.tasks.sentiment.models import (
    InteractionSentimentEvaluation,
    InteractionSentimentLabel,
    InteractionSnapshot,
)


@pytest.mark.asyncio
async def test_extract_recent_interactions_returns_ordered_snapshots():
    first_id = str(uuid4())
    second_id = str(uuid4())

    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_last_user_interaction_ids.return_value = [first_id, second_id]
    mock_graph_engine.get_nodes.return_value = [
        {
            "id": first_id,
            "question": "What is the status?",
            "answer": "All systems operational.",
            "context": "status check",
        },
        {
            "id": second_id,
            "question": "Need help",
            "answer": "Providing assistance",
            "context": "",
        },
    ]

    with patch(
        "cognee.tasks.sentiment.extract_recent_interactions.get_graph_engine",
        AsyncMock(return_value=mock_graph_engine),
    ):
        result = await extract_recent_interactions(None, last_k=5)

    assert len(result) == 2
    assert str(result[0].interaction_id) == first_id
    assert result[0].context == "status check"


@pytest.mark.asyncio
async def test_classify_interaction_sentiment_creates_datapoints():
    interaction = InteractionSnapshot(
        interaction_id=uuid4(),
        question="How was my experience?",
        answer="Glad you enjoyed it!",
        context="",
    )

    evaluation = InteractionSentimentEvaluation(
        sentiment=InteractionSentimentLabel.POSITIVE,
        confidence=0.92,
        summary="User expressed appreciation for the response.",
    )

    with patch(
        "cognee.tasks.sentiment.classify_interaction_sentiment.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=evaluation),
    ):
        datapoints = await classify_interaction_sentiment([interaction])

    assert len(datapoints) == 1
    sentiment = datapoints[0]
    assert sentiment.interaction_id == interaction.interaction_id
    assert sentiment.sentiment is InteractionSentimentLabel.POSITIVE
    assert sentiment.summary == evaluation.summary.strip()
    assert sentiment.metadata["index_fields"] == []


@pytest.mark.asyncio
async def test_link_sentiment_to_interactions_adds_edges():
    interaction_id = uuid4()

    with patch(
        "cognee.tasks.sentiment.classify_interaction_sentiment.LLMGateway.acreate_structured_output",
        AsyncMock(
            return_value=InteractionSentimentEvaluation(
                sentiment=InteractionSentimentLabel.NEGATIVE,
                confidence=0.21,
                summary="User is dissatisfied.",
            )
        ),
    ):
        sentiments = await classify_interaction_sentiment(
            [
                InteractionSnapshot(
                    interaction_id=interaction_id,
                    question="Why is this slow?",
                    answer="Investigating the delay.",
                    context="",
                )
            ]
        )

    mock_graph_engine = AsyncMock()
    with (
        patch(
            "cognee.tasks.sentiment.link_sentiment_to_interactions.get_graph_engine",
            AsyncMock(return_value=mock_graph_engine),
        ),
        patch(
            "cognee.tasks.sentiment.link_sentiment_to_interactions.index_graph_edges",
            AsyncMock(),
        ) as mock_index,
    ):
        result = await link_sentiment_to_interactions(sentiments)

    mock_graph_engine.add_edges.assert_awaited_once()
    edges = mock_graph_engine.add_edges.await_args.args[0]
    assert edges[0][0] == str(interaction_id)
    assert edges[0][2] == "has_sentiment"
    assert result == sentiments
    mock_index.assert_awaited_once()
