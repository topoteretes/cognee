from functools import lru_cache
from fastapi_users.authentication import AuthenticationBackend, JWTStrategy

from cognee.modules.users.authentication.api_key.get_api_key_transport import get_api_key_transport
from cognee.modules.users.authentication.api_key.api_key_jwt_strategy import ApiKeyJWTStrategy


@lru_cache
def get_api_key_backend():
    transport = get_api_key_transport()

    def get_jwt_strategy() -> JWTStrategy:
        return ApiKeyJWTStrategy()

    auth_backend = AuthenticationBackend(
        name=transport.name,  # type: ignore
        transport=transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend
