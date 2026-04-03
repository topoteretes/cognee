import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.UserApiKey import UserApiKey
from .exceptions import ApiKeyCreationError
from .get_api_keys import get_api_keys
from .hash_api_key import prepare_api_key


logger = get_logger(__name__)


class ApiKeySettings(BaseSettings):
    max_user_api_keys: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


apiKeySettings = ApiKeySettings()


async def create_api_key(user: User, name: str = None):
    existing_api_keys = await get_api_keys(user)

    if len(existing_api_keys) >= apiKeySettings.max_user_api_keys:
        raise ApiKeyCreationError("You have reached the maximum number of API keys.")

    relational_engine = get_relational_engine()

    api_key = generate_api_key()
    prepared_api_key = prepare_api_key(api_key)
    label = api_key[:8] + "****"

    async with relational_engine.get_async_session() as session:
        user_api_key = UserApiKey(
            user_id=user.id,
            api_key=prepared_api_key,
            label=label,
            name=name,
        )
        session.add(user_api_key)

        try:
            await session.commit()
            user_api_key.api_key = api_key  # return raw key so caller can show it once
            return user_api_key
        except Exception as error:
            logger.error(f"Failed to create API key for user {user.id}: {str(error)}")
            await session.rollback()
            raise ApiKeyCreationError("Failed to create API key, please try again.")


def generate_api_key():
    return secrets.token_hex(32)
