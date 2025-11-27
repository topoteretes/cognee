"""Integration test for interaction sentiment classification pipeline."""

import pathlib

import cognee
import pytest

from unittest.mock import AsyncMock, patch

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.pipelines.models import PipelineRunCompleted
from cognee.modules.search.types import SearchType
from cognee.tasks.sentiment.models import InteractionSentimentEvaluation, InteractionSentimentLabel


@pytest.mark.asyncio
async def test_interaction_sentiment_pipeline_creates_sentiment_nodes():
    data_path = pathlib.Path(__file__).parent / ".data_storage/test_interaction_sentiment"
    system_path = pathlib.Path(__file__).parent / ".cognee_system/test_interaction_sentiment"

    cognee.config.data_root_directory(str(data_path.resolve()))
    cognee.config.system_root_directory(str(system_path.resolve()))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "interaction_sentiment_test"

    await cognee.add("Cognee records conversations for later reasoning.", dataset_name)
    await cognee.cognify([dataset_name])

    await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Explain how you will remember this conversation.",
        save_interaction=True,
    )

    graph_engine = await get_graph_engine()
    nodes_before, _ = await graph_engine.get_graph_data()
    interaction_ids = {
        node_id for node_id, props in nodes_before if props.get("type") == "CogneeUserInteraction"
    }
    assert interaction_ids, "Expected at least one saved interaction before running memify"

    sentiment_response = InteractionSentimentEvaluation(
        sentiment=InteractionSentimentLabel.POSITIVE,
        confidence=0.88,
        summary="User is satisfied with the interaction and trusts Cognee.",
    )

    with patch(
        "cognee.tasks.sentiment.classify_interaction_sentiment.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=sentiment_response),
    ):
        result = await cognee.memify(dataset=dataset_name, sentiment_last_k=10)

    assert isinstance(result, PipelineRunCompleted), "Memify run did not complete successfully"

    nodes_after, edges_after = await graph_engine.get_graph_data()

    sentiment_nodes = [
        (node_id, props)
        for node_id, props in nodes_after
        if props.get("type") == "InteractionSentiment"
    ]

    assert sentiment_nodes, "Expected InteractionSentiment nodes to be created"

    for _, props in sentiment_nodes:
        assert props.get("summary"), "Sentiment node missing summary"
        assert props.get("sentiment") in {label.value for label in InteractionSentimentLabel}

    sentiment_node_ids = {node_id for node_id, _ in sentiment_nodes}
    linking_edges = [
        edge
        for edge in edges_after
        if edge[2] == "has_sentiment"
        and edge[0] in interaction_ids
        and edge[1] in sentiment_node_ids
    ]

    assert linking_edges, "Expected has_sentiment edges linking interactions to sentiments"

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
