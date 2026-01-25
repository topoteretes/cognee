"""
End-to-end integration test for feedback enrichment feature.

Tests the complete feedback enrichment pipeline:
1. Add data and cognify
2. Run search with save_interaction=True to create CogneeUserInteraction nodes
3. Submit feedback to create CogneeUserFeedback nodes
4. Run memify with feedback enrichment tasks to create FeedbackEnrichment nodes
5. Verify all nodes and edges are properly created and linked in the graph
"""

import os
import pathlib
from uuid import UUID, uuid4
from collections import Counter

from pydantic import BaseModel

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.feedback.create_enrichments import create_enrichments
from cognee.tasks.feedback.extract_feedback_interactions import (
    extract_feedback_interactions,
)
from cognee.tasks.feedback.generate_improved_answers import generate_improved_answers
from cognee.tasks.feedback.link_enrichments_to_feedback import (
    link_enrichments_to_feedback,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".data_storage/test_feedback_enrichment",
            )
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".cognee_system/test_feedback_enrichment",
            )
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "feedback_enrichment_test"

    await cognee.add("Cognee turns documents into AI memory.", dataset_name)
    await cognee.cognify([dataset_name])

    question_text = "Say something."
    result = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question_text,
        save_interaction=True,
    )

    assert len(result) > 0, "Search should return non-empty results"

    feedback_text = "This answer was completely useless, my feedback is definitely negative."
    await cognee.search(
        query_type=SearchType.FEEDBACK,
        query_text=feedback_text,
        last_k=1,
    )

    graph_engine = await get_graph_engine()
    nodes_before, edges_before = await graph_engine.get_graph_data()

    interaction_nodes_before = [
        (node_id, props)
        for node_id, props in nodes_before
        if props.get("type") == "CogneeUserInteraction"
    ]
    feedback_nodes_before = [
        (node_id, props)
        for node_id, props in nodes_before
        if props.get("type") == "CogneeUserFeedback"
    ]

    edge_types_before = Counter(edge[2] for edge in edges_before)

    assert len(interaction_nodes_before) >= 1, (
        f"Expected at least 1 CogneeUserInteraction node, found {len(interaction_nodes_before)}"
    )
    assert len(feedback_nodes_before) >= 1, (
        f"Expected at least 1 CogneeUserFeedback node, found {len(feedback_nodes_before)}"
    )

    for node_id, props in feedback_nodes_before:
        sentiment = props.get("sentiment", "")
        score = props.get("score", 0)
        feedback_text = props.get("feedback", "")
        logger.info(
            "Feedback node created",
            feedback=feedback_text,
            sentiment=sentiment,
            score=score,
        )

    assert edge_types_before.get("gives_feedback_to", 0) >= 1, (
        f"Expected at least 1 'gives_feedback_to' edge, found {edge_types_before.get('gives_feedback_to', 0)}"
    )

    extraction_tasks = [Task(extract_feedback_interactions, last_n=5)]
    enrichment_tasks = [
        Task(generate_improved_answers, top_k=20),
        Task(create_enrichments),
        Task(
            extract_graph_from_data,
            graph_model=KnowledgeGraph,
            task_config={"batch_size": 10},
        ),
        Task(add_data_points, task_config={"batch_size": 10}),
        Task(link_enrichments_to_feedback),
    ]

    class EnrichmentData(BaseModel):
        id: UUID

    await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[EnrichmentData(id=uuid4())],
        dataset=dataset_name,
    )

    nodes_after, edges_after = await graph_engine.get_graph_data()

    enrichment_nodes = [
        (node_id, props)
        for node_id, props in nodes_after
        if props.get("type") == "FeedbackEnrichment"
    ]

    assert len(enrichment_nodes) >= 1, (
        f"Expected at least 1 FeedbackEnrichment node, found {len(enrichment_nodes)}"
    )

    for node_id, props in enrichment_nodes:
        assert "text" in props, f"FeedbackEnrichment node {node_id} missing 'text' property"

    enrichment_node_ids = {node_id for node_id, _ in enrichment_nodes}
    edges_with_enrichments = [
        edge
        for edge in edges_after
        if edge[0] in enrichment_node_ids or edge[1] in enrichment_node_ids
    ]

    assert len(edges_with_enrichments) >= 1, (
        f"Expected enrichment nodes to have at least 1 edge, found {len(edges_with_enrichments)}"
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    logger.info("All feedback enrichment tests passed successfully")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
