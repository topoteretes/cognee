import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.UserApiKey import UserApiKey
from .exceptions import ApiKeyCreationError
from .get_api_keys import get_api_keys


logger = get_logger(__name__)


class ApiKeySettings(BaseSettings):
    max_user_api_keys: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


apiKeySettings = ApiKeySettings()


async def create_api_key(user: User):
    existing_api_keys = await get_api_keys(user)

    if len(existing_api_keys) >= apiKeySettings.max_user_api_keys:
        raise ApiKeyCreationError("You have reached the maximum number of API keys.")

    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        api_key = generate_api_key()
        user_api_key = UserApiKey(user_id=user.id, api_key=api_key)
        session.add(user_api_key)

        try:
            await session.commit()
            return user_api_key
        except Exception as error:
            logger.error(f"Failed to create API key for user {user.id}: {str(error)}")
            await session.rollback()
            raise ApiKeyCreationError("Failed to create API key, please try again.")


def generate_api_key():
    return secrets.token_hex(24)
