import logging
from typing import Union

from pydantic import BaseModel
from cognee.shared.utils import setup_logging
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search
from cognee.tasks.summarization import summarize_text

from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph.extract_graph_from_data_chunks import extract_graph_from_data_chunks

from cognee.tasks.storage import add_data_points
from cognee.api.v1.cognify.cognify_v2 import cognify


logger = logging.getLogger("cognify.25q1")


async def get_25q1_tasks(user: User = None, summaries=True) -> list[Task]:
    """Get the task list for 25Q1 pipeline focusing on entity extraction."""
    if user is None:
        user = await get_default_user()

    try:
        default_tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user=user, permissions=["write"]),
            Task(extract_chunks_from_documents, max_chunk_tokens=get_max_chunk_tokens()),
            Task(extract_graph_from_data_chunks, n_rounds=2, task_config={"batch_size": 10}),
        ]
        if summaries:
            default_tasks.append(
                Task(
                    summarize_text,
                    summarization_model=get_cognify_config().summarization_model,
                    task_config={"batch_size": 10},
                )
            )
        default_tasks.append(
            Task(add_data_points, task_config={"batch_size": 10}),
        )
    except Exception as error:
        logger.error("Failed to create 25Q1 tasks: %s", error)
        raise error
    return default_tasks


async def cognify_25q1(
    datasets: Union[str, list[str]] = None,
    user: User = None,
):
    """Run the 25Q1 version of cognify focusing on multi-step graph extraction."""
    tasks = await get_25q1_tasks(user)
    return await cognify(tasks=tasks)


if __name__ == "__main__":
    import asyncio
    import cognee

    async def main():
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        text = """
        Natural language processing (NLP) is an interdisciplinary
        subfield of computer science and information retrieval.
        """

        await cognee.add(text)
        await cognify_25q1()

    asyncio.run(main())
