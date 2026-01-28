from uuid import UUID

from cognee import memify

from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.data_models import KnowledgeGraph

from cognee.tasks.feedback.extract_feedback_interactions import extract_feedback_interactions
from cognee.tasks.feedback.generate_improved_answers import generate_improved_answers
from cognee.tasks.feedback.create_enrichments import create_enrichments
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.feedback.link_enrichments_to_feedback import link_enrichments_to_feedback


async def feedback_enrichment_memify(dataset: str | UUID | None = None, last_n: int = 5):
    """Execute memify with extraction, answer improvement, enrichment creation, and graph processing tasks."""

    kwargs = {}
    if dataset is not None:
        kwargs["dataset"] = dataset

    # Instantiate tasks with their own kwargs
    extraction_tasks = [Task(extract_feedback_interactions, last_n=last_n)]
    enrichment_tasks = [
        Task(generate_improved_answers, top_k=20),
        Task(create_enrichments),
        Task(extract_graph_from_data, graph_model=KnowledgeGraph, task_config={"batch_size": 10}),
        Task(add_data_points, task_config={"batch_size": 10}),
        Task(link_enrichments_to_feedback),
    ]
    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # A placeholder to prevent fetching the entire graph
        **kwargs,
    )

    return result
