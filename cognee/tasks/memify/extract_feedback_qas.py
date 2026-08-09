from typing import List, Optional

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.cache import SessionQAEntry
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)

logger = get_logger("extract_feedback_qas")


def _is_eligible(entry: SessionQAEntry) -> bool:
    feedback_score = entry.feedback_score
    if not isinstance(feedback_score, int) or feedback_score < 1 or feedback_score > 5:
        return False

    memify_metadata = entry.memify_metadata
    if (
        isinstance(memify_metadata, dict)
        and memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY) is True
    ):
        return False

    # Entries without usable graph element ids are still yielded once: the apply task
    # marks them processed so they are never rescanned on later improve() runs.
    return True


async def extract_feedback_qas(data, session_ids: Optional[List[str]] = None):
    """
    Read provided sessions and yield feedback QAs eligible for graph weight updates.
    """
    if (
        not isinstance(session_ids, list)
        or not session_ids
        or any(not isinstance(session_id, str) or not session_id for session_id in session_ids)
    ):
        raise CogneeValidationError(
            message="session_ids must be provided for extract_feedback_qas",
            log=False,
        )

    if not data or data == [{}]:
        logger.info("Extracting feedback QAs from session cache")

    user: User = session_user.get()
    if not user:
        raise CogneeSystemError(message="No authenticated user found in context", log=False)

    session_manager = get_session_manager()

    user_id = str(user.id)

    for session_id in session_ids:
        entries = await session_manager.get_session(
            user_id=user_id,
            session_id=session_id,
            formatted=False,
        )
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, SessionQAEntry) or not _is_eligible(entry):
                continue

            qa_id = entry.qa_id
            if not isinstance(qa_id, str) or not qa_id:
                continue

            memify_metadata = entry.memify_metadata
            yield {
                "session_id": session_id,
                "qa_id": qa_id,
                "feedback_score": entry.feedback_score,
                "used_graph_element_ids": entry.used_graph_element_ids,
                "memify_metadata": memify_metadata if isinstance(memify_metadata, dict) else {},
            }
