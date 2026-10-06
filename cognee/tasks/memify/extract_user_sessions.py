from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_node_set import (
    node_set_from_rows,
    read_session_context_strict,
)
from cognee.infrastructure.session.session_persist_watermark import (
    SESSION_PERSIST_WATERMARK,
    SessionPersistWindow,
)
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("extract_user_sessions")


async def extract_user_sessions(
    data,
    session_ids: list[str] | None = None,
):
    """
    Extract not-yet-persisted Q&A entries for the current user via SessionManager.

    For each session, reads the persist watermark (see
    ``session_persist_watermark``) and yields ONE window containing only the
    Q&A entries above it, formatted as question/answer text. A session with no
    new entries yields nothing, so re-running improve() on an unchanged
    session does zero ingestion work. The watermark itself is advanced by
    ``cognify_session`` only after the window is successfully cognified.

    Args:
        data: Data passed from memify. If empty dict ({}), no external data is provided.
        session_ids: Optional list of specific session IDs to extract.

    Yields:
        SessionPersistWindow covering the session's unpersisted entries.

    Raises:
        CogneeSystemError: If SessionManager is unavailable or extraction fails.
    """
    try:
        if not data or data == [{}]:
            logger.info("Fetching session metadata for current user")

        user: User = session_user.get()
        if not user:
            raise CogneeSystemError(message="No authenticated user found in context", log=False)

        user_id = str(user.id)

        session_manager = get_session_manager()
        if not session_manager.is_available:
            raise CogneeSystemError(
                message="SessionManager not available for session extraction, please enable caching in order to have sessions to save",
                log=False,
            )

        if session_ids:
            for session_id in session_ids:
                try:
                    qa_data = await session_manager.get_session(
                        user_id=user_id,
                        session_id=session_id,
                        formatted=False,
                    )
                    if not qa_data:
                        continue

                    # One strict snapshot of the session's context rows holds both
                    # the persist watermark and the pinned node set. It is taken
                    # after the Q&A read, so every entry in qa_data was written
                    # before it, and a pin written ahead of any of them is in it.
                    # Strict: an unreadable session is skipped for this run (the
                    # except below), never bridged from watermark 0 or untagged.
                    context_rows = await read_session_context_strict(
                        session_manager, user_id, session_id
                    )
                    persisted_count = SESSION_PERSIST_WATERMARK.resolve_effective(
                        SESSION_PERSIST_WATERMARK.count_from_rows(context_rows),
                        len(qa_data),
                        session_id=session_id,
                    )

                    new_entries = qa_data[persisted_count:]
                    if not new_entries:
                        logger.info(
                            "Session %s already persisted up to entry %d, nothing new",
                            session_id,
                            persisted_count,
                        )
                        continue

                    logger.info(
                        "Extracted session %s via SessionManager: %d new of %d total Q&A pairs",
                        session_id,
                        len(new_entries),
                        len(qa_data),
                    )
                    session_string = f"Session ID: {session_id}\n\n"
                    for qa_pair in new_entries:
                        question = qa_pair.question
                        answer = qa_pair.answer
                        session_string += f"Question: {question}\n\nAnswer: {answer}\n\n"

                    yield SessionPersistWindow(
                        user_id=user_id,
                        session_id=session_id,
                        text=session_string,
                        persisted_qa_count=len(qa_data),
                        node_set=node_set_from_rows(context_rows),
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract session {session_id}: {e!s}", exc_info=True)
                    continue
        else:
            logger.info(
                "No specific session_ids provided. Please specify which sessions to extract."
            )

    except CogneeSystemError:
        raise
    except Exception as e:
        logger.exception("Error extracting user sessions")
        raise CogneeSystemError(message=f"Failed to extract user sessions: {e!s}", log=False)


async def has_new_session_qa(session_manager, user_id: str, session_ids: list[str]) -> bool:
    """Whether any session holds Q&A entries the persist watermark hasn't covered.

    The improve stage's pre-check: when False, the stage reports
    ``already_completed`` without running the memify pipeline at all — an
    unconditional pipeline run logs a completed ``memify_pipeline`` row even
    with nothing new, which the enrichment change-check would count as a graph
    write. Mirrors the extraction loop above, including its stale-watermark
    rule (a watermark above the entry count means the session was rebuilt:
    everything is pending).
    """
    for session_id in session_ids:
        qa_data = await session_manager.get_session(
            user_id=user_id, session_id=session_id, formatted=False
        )
        if not qa_data:
            continue
        persisted_count = await SESSION_PERSIST_WATERMARK.read_count(
            session_manager, user_id, session_id
        )
        if persisted_count != len(qa_data):
            return True
    return False
