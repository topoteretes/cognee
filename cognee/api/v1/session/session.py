from typing import List, Optional

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.exceptions.exceptions import UserNotFoundError
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_api_sdk")


async def _resolve_user(user: Optional[User]) -> User:
    if user is not None:
        if getattr(user, "id", None) is None:
            raise CogneeValidationError(
                message="Session user must have an id.",
                name="SessionPreconditionError",
            )
        return user
    ctx_user = session_user.get()
    if ctx_user is not None and getattr(ctx_user, "id", None) is not None:
        return ctx_user
    try:
        return await get_default_user()
    except (DatabaseNotCreatedError, UserNotFoundError) as error:
        raise CogneeValidationError(
            message=(
                "Session prerequisites not met: no default user found. "
                "Initialize Cognee before using session APIs by running "
                "`await cognee.add(...)` followed by `await cognee.cognify()`."
            ),
            name="SessionPreconditionError",
        ) from error


async def get_session(
    session_id: str = "default_session",
    last_n: Optional[int] = None,
    user: Optional[User] = None,
) -> List[SessionQAEntry]:
    resolved_user = await _resolve_user(user)
    user_id = str(resolved_user.id)

    try:
        sm = get_session_manager()
        raw = await sm.get_session(
            user_id=user_id,
            session_id=session_id,
            last_n=last_n,
            formatted=False,
        )
    except Exception as e:
        logger.warning("get_session: error from SessionManager: %s", e)
        return []

    if not raw:
        return []

    result: List[SessionQAEntry] = []
    for entry in raw:
        if isinstance(entry, dict):
            try:
                result.append(SessionQAEntry.model_validate(entry))
            except Exception as e:
                logger.error("get_session: skip invalid entry: %s", e)
        else:
            result.append(entry)
    return result
