import os
from functools import lru_cache
from typing import Optional

from fastapi_users.authentication import AuthenticationBackend

from cognee.modules.users.authentication.api_bearer import api_bearer_transport
from .jwks_jwt_strategy import JWKSJWTStrategy


@lru_cache
def get_jwks_auth_backend() -> Optional[AuthenticationBackend]:
    """
    Returns the JWKS authentication backend if COGNEE_JWKS_URL is set.
    Otherwise, returns None.
    """
    if not os.getenv("COGNEE_JWKS_URL"):
        return None

    def get_strategy() -> JWKSJWTStrategy:
        return JWKSJWTStrategy()

    auth_backend = AuthenticationBackend(
        name="jwks_bearer",
        transport=api_bearer_transport,
        get_strategy=get_strategy,
    )

    return auth_backend
