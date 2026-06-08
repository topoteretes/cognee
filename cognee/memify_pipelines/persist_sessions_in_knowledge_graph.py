from typing import Optional, List

from cognee import memify
from cognee.api.v1.cognify.cognify import get_cognify_processing_tasks
from cognee.context_global_variables import (
    set_database_global_context_variables,
    set_session_user_context_variable,
)
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.operations.worker_pipeline import FixedWorkers
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify import extract_user_sessions
from cognee.tasks.memify.save_session_as_data import save_session_as_data


logger = get_logger("persist_sessions_in_knowledge_graph")

SESSION_NODE_SET = ["user_sessions_from_cache"]


async def persist_sessions_in_knowledge_graph_pipeline(
    user: User,
    session_ids: Optional[List[str]] = None,
    dataset: str = "main_dataset",
    run_in_background: bool = False,
):
    """
    Persist user sessions into the knowledge graph via memify pipeline.

    Reads session data via SessionManager (caching must be enabled). Each
    session is saved as a Data row tagged with ``node_set=["user_sessions_from_cache"]``
    and streamed through cognify's standard per-document stages
    (classify → chunk → extract → add_data_points) in a single pipeline run.

    Earlier implementations nested ``cognee.cognify()`` calls inside the
    enrichment task, which raced when multiple sessions targeted the same
    dataset. The fused pipeline below processes each session as one item
    flowing through the worker queues, so parallelism is across distinct
    Data items (safe — same as a regular ``cognee.cognify()``) rather than
    across overlapping dataset-wide cognify runs (not safe).

    Args:
        user: Authenticated user with write access to the dataset.
        session_ids: Optional list of session IDs to persist. If None, no
            sessions are extracted (caller must specify which sessions to
            persist).
        dataset: Dataset name for write access. Defaults to "main_dataset".
        run_in_background: If True, runs memify asynchronously and returns
            immediately.
    """
    await set_session_user_context_variable(user)
    dataset_to_write = await get_authorized_existing_datasets(
        user=user, datasets=[dataset], permission_type="write"
    )

    if not dataset_to_write:
        raise CogneeValidationError(
            message=f"User (id: {str(user.id)}) does not have write access to dataset: {dataset}",
            log=False,
        )

    target_dataset = dataset_to_write[0]

    async with set_database_global_context_variables(target_dataset.id, target_dataset.owner_id):
        extraction_tasks = [Task(extract_user_sessions, session_ids=session_ids)]

        # ``save_session_as_data`` is pinned to a single worker so concurrent
        # session writes don't race on the dataset's Data rows. Downstream
        # stages (the cognify pipeline) are safe to parallelize across
        # distinct Data items.
        enrichment_tasks = [
            Task(
                save_session_as_data,
                dataset_id=target_dataset.id,
                user=user,
                node_set=SESSION_NODE_SET,
                workers=FixedWorkers(1),
            ),
            *(await get_cognify_processing_tasks(user=user)),
        ]

        result = await memify(
            extraction_tasks=extraction_tasks,
            enrichment_tasks=enrichment_tasks,
            dataset=target_dataset.id,
            data=[{}],
            run_in_background=run_in_background,
        )

    logger.info("Session persistence pipeline completed")
    return result
