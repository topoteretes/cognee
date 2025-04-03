from typing import List
from pydantic import BaseModel

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.operations.needs import merge_needs
from cognee.modules.pipelines.tasks import Task, TaskConfig
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph.extract_graph_from_data_v2 import (
    extract_graph_from_data,
)
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.infrastructure.llm import get_max_chunk_tokens


async def get_cascade_graph_tasks(
    user: User = None, graph_model: BaseModel = KnowledgeGraph
) -> List[Task]:
    """Retrieve cascade graph tasks asynchronously."""
    if user is None:
        user = await get_default_user()

    cognee_config = get_cognify_config()
    default_tasks = [
        Task(classify_documents),
        Task(
            check_permissions_on_documents,
            user=user,
            permissions=["write"],
            task_config=TaskConfig(needs=[classify_documents]),
        ),
        Task(  # Extract text chunks based on the document type.
            extract_chunks_from_documents,
            max_chunk_tokens=get_max_chunk_tokens(),
            task_config=TaskConfig(needs=[check_permissions_on_documents], output_batch_size=50),
        ),
        Task(
            extract_graph_from_data,
            task_config=TaskConfig(needs=[extract_chunks_from_documents]),
        ),  # Generate knowledge graphs using cascade extraction
        Task(
            summarize_text,
            summarization_model=cognee_config.summarization_model,
            task_config=TaskConfig(needs=[extract_chunks_from_documents]),
        ),
        Task(
            add_data_points,
            task_config=TaskConfig(needs=[merge_needs(summarize_text, extract_graph_from_data)]),
        ),
    ]
    return default_tasks
