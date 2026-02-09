from typing import Optional, List

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User

logger = get_logger("extract_user_sessions")


async def extract_user_sessions(
    data,
    session_ids: Optional[List[str]] = None,
):
    """
    Extract Q&A sessions for the current user via SessionManager.

    Retrieves all Q&A triplets from specified session IDs and yields them
    as formatted strings combining question, context, and answer. Session
    persistence relies on SessionManager; caching must be enabled for
    sessions to be available.

    Args:
        data: Data passed from memify. If empty dict ({}), no external data is provided.
        session_ids: Optional list of specific session IDs to extract.

    Yields:
        String containing session ID and all Q&A pairs formatted.

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
                    if qa_data:
                        logger.info(
                            f"Extracted session {session_id} via SessionManager with {len(qa_data)} Q&A pairs"
                        )
                        session_string = f"Session ID: {session_id}\n\n"
                        for qa_pair in qa_data:
                            question = qa_pair.get("question", "")
                            answer = qa_pair.get("answer", "")
                            session_string += f"Question: {question}\n\nAnswer: {answer}\n\n"
                        yield session_string
                except Exception as e:
                    logger.warning(f"Failed to extract session {session_id}: {str(e)}")
                    continue
        else:
            logger.info(
                "No specific session_ids provided. Please specify which sessions to extract."
            )

    except CogneeSystemError:
        raise
    except Exception as e:
        logger.error(f"Error extracting user sessions: {str(e)}")
        raise CogneeSystemError(message=f"Failed to extract user sessions: {str(e)}", log=False)
