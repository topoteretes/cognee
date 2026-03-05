from typing import List

from cognee import memify
from cognee.context_global_variables import (
    set_database_global_context_variables,
    set_session_user_context_variable,
)
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.apply_feedback_weights import apply_feedback_weights
from cognee.tasks.memify.extract_feedback_qas import extract_feedback_qas

logger = get_logger("apply_feedback_weights_pipeline")


async def apply_feedback_weights_pipeline(
    user: User,
    session_ids: List[str],
    dataset: str = "main_dataset",
    alpha: float = 0.1,
    batch_size: int = 100,
    run_in_background: bool = False,
):
    """
    Apply session feedback-based graph feedback_weight updates via memify pipeline.

    This pipeline reads QAs from provided session IDs, filters eligible feedback entries,
    updates mapped graph nodes/edges with streaming feedback weights, and marks each QA
    as applied in memify metadata only on full success.
    """
    if (
        not isinstance(session_ids, list)
        or not session_ids
        or any(not isinstance(session_id, str) or not session_id for session_id in session_ids)
    ):
        raise CogneeValidationError(message="session_ids must be a non-empty list", log=False)

    if alpha <= 0 or alpha > 1:
        raise CogneeValidationError(message="alpha must be in range (0, 1]", log=False)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise CogneeValidationError(message="batch_size must be a positive integer", log=False)

    await set_session_user_context_variable(user)

    dataset_to_write = await get_authorized_existing_datasets(
        user=user,
        datasets=[dataset],
        permission_type="write",
    )
    if not dataset_to_write:
        raise CogneeValidationError(
            message=f"User (id: {str(user.id)}) does not have write access to dataset: {dataset}",
            log=False,
        )

    await set_database_global_context_variables(
        dataset_to_write[0].id, dataset_to_write[0].owner_id
    )

    extraction_tasks = [Task(extract_feedback_qas, session_ids=session_ids)]
    enrichment_tasks = [
        Task(apply_feedback_weights, alpha=alpha, task_config={"batch_size": batch_size})
    ]

    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset_to_write[0].id,
        data=[{}],
        user=user,
        run_in_background=run_in_background,
    )

    logger.info(
        "Feedback weight memify pipeline completed",
        extra={
            "dataset_id": str(dataset_to_write[0].id),
            "session_count": len(session_ids),
            "alpha": alpha,
            "batch_size": batch_size,
        },
    )

    return result
