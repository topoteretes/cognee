from typing import List
from pydantic import BaseModel

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.utils import send_telemetry
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

    try:
        cognee_config = get_cognify_config()
        default_tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user=user, permissions=["write"]),
            Task(
                extract_chunks_from_documents, max_chunk_tokens=get_max_chunk_tokens()
            ),  # Extract text chunks based on the document type.
            Task(
                extract_graph_from_data, task_config={"batch_size": 10}
            ),  # Generate knowledge graphs using cascade extraction
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config={"batch_size": 10},
            ),
            Task(add_data_points, task_config={"batch_size": 10}),
        ]
    except Exception as error:
        send_telemetry("cognee.cognify DEFAULT TASKS CREATION ERRORED", user.id)
        raise error
    return default_tasks
