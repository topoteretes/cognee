import re
from datetime import datetime, timezone
from uuid import UUID

import cognee
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import (
    SessionPersistWindow,
    save_persisted_qa_count,
)
from cognee.modules.improve.constants import USER_SESSIONS_NODE_SET
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

logger = get_logger("cognify_session")

SESSION_MEMORY_FILE_TAG = "session-memory"
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def session_memory_filename(session_id: str, synced_at: datetime | None = None) -> str:
    """``<utc time>_session-memory_<session id>.txt`` for a persisted session window.

    Under the content-hash default (``text_<md5>.txt``) a dataset's file list
    could not say which session a document came from or when it was bridged;
    the name now carries both, and the tag lets readers pick these files out.
    Characters a file system or URL would choke on are folded to ``-``.
    """
    moment = (synced_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    safe_session_id = _UNSAFE_FILENAME_CHARS.sub("-", session_id).strip("-") or "session"
    return f"{moment:%Y-%m-%dT%H-%M-%SZ}_{SESSION_MEMORY_FILE_TAG}_{safe_session_id}.txt"


async def cognify_session(
    data: SessionPersistWindow | list[SessionPersistWindow],
    dataset_id: UUID | str | None = None,
    user: User | None = None,
) -> None:
    """
    Cognify session windows into the knowledge graph and advance their watermarks.

    Receives one ``SessionPersistWindow`` (or a batch of them — the pipeline
    runner delivers generator output in batches) from ``extract_user_sessions``.
    For each window: adds its text to cognee with the
    ``USER_SESSIONS_NODE_SET`` node set, triggers cognify, and — only after
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
                DataItem(data=window.text, name=session_memory_filename(window.session_id)),
                dataset_id=dataset_id,
                node_set=[USER_SESSIONS_NODE_SET],
                user=user,
            )
            logger.debug("Session data added to cognee with node_set: %s", USER_SESSIONS_NODE_SET)
            # raise_on_error=False: one window's failed build must not kill the
            # whole memify run — inspect the run info instead, keep this
            # window's watermark put (so it is re-extracted and retried on the
            # next improve()), and continue with the remaining windows.
            cognify_result = await cognee.cognify(
                datasets=[dataset_id], user=user, raise_on_error=False
            )
            errored_run = get_errored_run_info(cognify_result)
            if errored_run is not None:
                logger.error(
                    "Cognify failed for session %s window (%s: %s); watermark not advanced, "
                    "window will be retried on the next improve()",
                    window.session_id,
                    errored_run.error_class,
                    errored_run.error_message,
                )
                continue
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
        logger.exception("Error cognifying session data")
        raise CogneeSystemError(message=f"Failed to cognify session data: {e!s}", log=False)
