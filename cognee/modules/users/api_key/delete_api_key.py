import secrets
from uuid import UUID
from sqlalchemy import select

from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.UserApiKey import UserApiKey
from .exceptions import ApiKeyDeletionError


logger = get_logger(__name__)


async def delete_api_key(user: User, api_key_id: UUID):
    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        try:
            user_api_key = (
                await session.execute(select(UserApiKey).filter_by(id=api_key_id, user_id=user.id))
            ).scalar()

            if not user_api_key:
                raise ApiKeyDeletionError(f"No API key found for user {user.id}.")

            await session.delete(user_api_key)

            await session.commit()
        except Exception as error:
            logger.error(f"Failed to delete API key for user {user.id}: {str(error)}")
            await session.rollback()

            raise ApiKeyDeletionError(f"Failed to delete API key for user {user.id}.")
