from typing import Optional

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
from cognee.tasks.memify import (
    cognify_agent_trace_feedback,
    extract_agent_trace_feedbacks,
)

logger = get_logger("persist_agent_trace_feedbacks_in_knowledge_graph")


async def persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
    user: User,
    session_ids: Optional[list[str]] = None,
    dataset: str = "main_dataset",
    node_set_name: str = "agent_trace_feedbacks",
    raw_trace_content: bool = False,
    last_n_steps: Optional[int] = None,
    run_in_background: bool = False,
):
    """
    Persist agent trace content into the knowledge graph via memify pipeline.

    Reads either per-step ``session_feedback`` values or raw ``method_return_value``
    values via SessionManager. Each session's non-empty entries are concatenated into
    one text blob and cognified into the graph with the provided node-set name.

    Args:
        user: Authenticated user with write access to the dataset.
        session_ids: Optional list of session IDs to persist. If None, no sessions
            are extracted (caller must specify which sessions to persist).
        dataset: Dataset name for write access. Defaults to "main_dataset".
        node_set_name: Node-set name used when adding the persisted feedback.
        raw_trace_content: When True, persist raw ``method_return_value`` values instead
            of ``session_feedback`` summaries.
        last_n_steps: Optional number of most recent trace steps to persist per
            session. When None, all stored steps are persisted.
        run_in_background: If True, runs memify asynchronously and returns immediately.
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

    await set_database_global_context_variables(
        dataset_to_write[0].id, dataset_to_write[0].owner_id
    )

    extraction_tasks = [
        Task(
            extract_agent_trace_feedbacks,
            session_ids=session_ids,
            raw_trace_content=raw_trace_content,
            last_n_steps=last_n_steps,
        )
    ]
    enrichment_tasks = [
        Task(
            cognify_agent_trace_feedback,
            dataset_id=dataset_to_write[0].id,
            node_set_name=node_set_name,
        ),
    ]

    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset_to_write[0].id,
        data=[{}],
        run_in_background=run_in_background,
    )

    logger.info(
        "Agent trace feedback persistence pipeline completed",
        extra={
            "dataset_id": str(dataset_to_write[0].id),
            "session_count": len(session_ids) if session_ids else 0,
            "node_set_name": node_set_name,
            "raw_trace_content": raw_trace_content,
            "last_n_steps": last_n_steps,
        },
    )
    return result
