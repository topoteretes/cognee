from typing import Optional, Union
from uuid import UUID

import cognee

from cognee.exceptions import CogneeValidationError, CogneeSystemError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import (
    SessionPersistWindow,
    save_persisted_qa_count,
)
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User

logger = get_logger("cognify_session")


async def cognify_session(
    data: Union[SessionPersistWindow, list[SessionPersistWindow]],
    dataset_id: Optional[UUID | str] = None,
    user: Optional[User] = None,
) -> None:
    """
    Cognify session windows into the knowledge graph and advance their watermarks.

    Receives one ``SessionPersistWindow`` (or a batch of them — the pipeline
    runner delivers generator output in batches) from ``extract_user_sessions``.
    For each window: adds its text to cognee with the
    "user_sessions_from_cache" node set, triggers cognify, and — only after
    both succeed — advances that session's persist watermark to the entry
    count captured at extraction time. On failure the watermark stays put, so
    the same window is re-extracted and retried on the next improve()
    (add-level content-hash dedup makes the retry safe).

    Args:
        data: Window(s) yielded by ``extract_user_sessions``.
        dataset_id: Dataset to cognify into.
        user: Authenticated user owning the sessions.

    Raises:
        CogneeValidationError: If no valid, non-empty window was provided.
        CogneeSystemError: If cognee operations fail.
    """
    windows = data if isinstance(data, list) else [data]
    valid_windows = [
        window
        for window in windows
        if isinstance(window, SessionPersistWindow) and window.text.strip()
    ]
    if not valid_windows:
        logger.warning("No session windows provided to cognify_session task, skipping")
        raise CogneeValidationError(message="Session window cannot be empty", log=False)

    try:
        for window in valid_windows:
            logger.info(
                "Processing session %s window (%d entries persisted after this) for cognification",
                window.session_id,
                window.persisted_qa_count,
            )

            await cognee.add(
                window.text,
                dataset_id=dataset_id,
                node_set=["user_sessions_from_cache"],
                user=user,
            )
            logger.debug("Session data added to cognee with node_set: user_sessions")
            await cognee.cognify(datasets=[dataset_id], user=user)
            logger.info("Session data successfully cognified")

            await save_persisted_qa_count(
                get_session_manager(),
                user_id=window.user_id,
                session_id=window.session_id,
                persisted_qa_count=window.persisted_qa_count,
            )
            logger.info(
                "Session %s persist watermark advanced to %d",
                window.session_id,
                window.persisted_qa_count,
            )

    except Exception as e:
        logger.error(f"Error cognifying session data: {str(e)}")
        raise CogneeSystemError(message=f"Failed to cognify session data: {str(e)}", log=False)
