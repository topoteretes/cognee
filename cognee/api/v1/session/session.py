from typing import List, Optional

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_api")


async def get_session(
    session_id: str = "default_session",
    last_n: Optional[int] = None,
    user: Optional[User] = None,
) -> List[SessionQAEntry]:

    logger.info(
        "get_session: session_id=%s, last_n=%s, user=%s -> returning []",
        session_id,
        last_n,
        getattr(user, "id", user),
    )
    return []


async def add_feedback(
    session_id: str,
    qa_id: str,
    feedback_text: Optional[str] = None,
    feedback_score: Optional[int] = None,
    user: Optional[User] = None,
) -> bool:

    logger.info(
        "add_feedback: session_id=%s, qa_id=%s, feedback_text=%s, feedback_score=%s, user=%s -> returning False",
        session_id,
        qa_id,
        feedback_text,
        feedback_score,
        getattr(user, "id", user),
    )
    return False
