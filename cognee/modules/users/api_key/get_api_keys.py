from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User
from cognee.modules.users.models.UserApiKey import UserApiKey
from cognee.shared.logging_utils import get_logger

from .exceptions import ApiKeyQueryError

logger = get_logger(__name__)


async def get_api_keys(user: User):
    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        try:
            user_api_keys = (
                (await session.execute(select(UserApiKey).where(UserApiKey.user_id == user.id)))
                .scalars()
                .all()
            )

            return user_api_keys
        except Exception:
            logger.exception(f"Failed to get API keys for user {user.id}")

            raise ApiKeyQueryError(f"Failed to get API keys for user {user.id}.")
