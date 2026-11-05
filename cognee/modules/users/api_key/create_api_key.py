import secrets
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User
from cognee.modules.users.models.UserApiKey import UserApiKey
from cognee.shared.logging_utils import get_logger

from .exceptions import ApiKeyCreationError
from .get_api_keys import get_api_keys
from .hash_api_key import prepare_api_key

logger = get_logger(__name__)


class ApiKeySettings(BaseSettings):
    max_user_api_keys: int = 10

    model_config = SettingsConfigDict(extra="allow")


apiKeySettings = ApiKeySettings()


def build_api_key_row(user_id: UUID, name: str | None) -> tuple[UserApiKey, str]:
    """A new, unsaved key row for ``user_id`` and the raw key it stores.

    The row holds the key as stored (hashed when ``HASH_API_KEY`` is on, which
    is slow: build it outside any lock). Every key is built here, so user keys
    and plugin keys share one label format and hashing scheme.
    """
    api_key = generate_api_key()
    user_api_key = UserApiKey(
        user_id=user_id,
        api_key=prepare_api_key(api_key),
        label=api_key[:8] + "****",
        name=name,
    )
    return user_api_key, api_key


async def create_api_key(user: User, name: str | None = None):
    existing_api_keys = await get_api_keys(user)

    if len(existing_api_keys) >= apiKeySettings.max_user_api_keys:
        raise ApiKeyCreationError("You have reached the maximum number of API keys.")

    relational_engine = get_relational_engine()

    user_api_key, api_key = build_api_key_row(user.id, name)

    async with relational_engine.get_async_session() as session:
        session.add(user_api_key)

        try:
            await session.commit()
            user_api_key.api_key = api_key  # return raw key so caller can show it once
            return user_api_key
        except Exception:
            logger.exception(f"Failed to create API key for user {user.id}")
            await session.rollback()
            raise ApiKeyCreationError("Failed to create API key, please try again.")


def generate_api_key():
    return secrets.token_hex(32)
