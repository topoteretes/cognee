from typing import Optional, List

from cognee import memify
from cognee.context_global_variables import (
    set_database_global_context_variables,
    set_session_user_context_variable,
)
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.tasks.memify import extract_user_sessions, cognify_session


logger = get_logger("persist_sessions_in_knowledge_graph")


async def persist_sessions_in_knowledge_graph_pipeline(
    user: User,
    session_ids: Optional[List[str]] = None,
    dataset: str = "main_dataset",
    run_in_background: bool = False,
):
    await set_session_user_context_variable(user)
    dataset_to_write = await get_authorized_existing_datasets(
        user=user, datasets=[dataset], permission_type="write"
    )

    if not dataset_to_write:
        raise CogneeValidationError(
            message=f"User (id: {str(user.id)}) does not have write access to dataset: {dataset}",
            log=False,
        )

    await set_database_global_context_variables(
        dataset_to_write[0].id, dataset_to_write[0].owner_id
    )

    extraction_tasks = [Task(extract_user_sessions, session_ids=session_ids)]

    enrichment_tasks = [
        Task(cognify_session, dataset_id=dataset_to_write[0].id),
    ]

    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset_to_write[0].id,
        data=[{}],
        run_in_background=run_in_background,
    )

    logger.info("Session persistence pipeline completed")
    return result
