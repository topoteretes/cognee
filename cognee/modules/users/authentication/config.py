from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthConfig(BaseSettings):
    fastapi_users_jwt_secret: str = "super_secret"
    fastapi_users_verification_token_secret: str = "super_secret"
    fastapi_users_reset_password_token_secret: str = "super_secret"
    jwt_lifetime_seconds: int = 3600
    hash_api_key: bool = False
    require_authentication: Optional[bool] = None
    enable_backend_access_control: bool = True
    accept_local_file_path: bool = True
    allow_http_requests: bool = True
    allow_cypher_query: bool = True
    raise_incremental_loading_errors: bool = True
    enable_last_accessed: bool = False
    auth_token_cookie_name: str = "auth_token"
    auth_token_cookie_domain: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "fastapi_users_jwt_secret": self.fastapi_users_jwt_secret,
            "fastapi_users_verification_token_secret": (
                self.fastapi_users_verification_token_secret
            ),
            "fastapi_users_reset_password_token_secret": (
                self.fastapi_users_reset_password_token_secret
            ),
            "jwt_lifetime_seconds": self.jwt_lifetime_seconds,
            "hash_api_key": self.hash_api_key,
            "require_authentication": self.require_authentication,
            "enable_backend_access_control": self.enable_backend_access_control,
            "accept_local_file_path": self.accept_local_file_path,
            "allow_http_requests": self.allow_http_requests,
            "allow_cypher_query": self.allow_cypher_query,
            "raise_incremental_loading_errors": self.raise_incremental_loading_errors,
            "enable_last_accessed": self.enable_last_accessed,
            "auth_token_cookie_name": self.auth_token_cookie_name,
            "auth_token_cookie_domain": self.auth_token_cookie_domain,
        }


@lru_cache
def get_auth_config():
    return AuthConfig()


def resolve_auth_posture() -> tuple[bool, bool, str]:
    auth_config = get_auth_config()
    enable_access_control = auth_config.enable_backend_access_control

    if auth_config.require_authentication is not None:
        require_authentication = auth_config.require_authentication
        if enable_access_control and not require_authentication:
            reason = "forced on by multi-tenant mode (REQUIRE_AUTHENTICATION=false was ignored)"
            return True, enable_access_control, reason
        return require_authentication, enable_access_control, "explicit REQUIRE_AUTHENTICATION"

    require_authentication = enable_access_control
    return (
        require_authentication,
        enable_access_control,
        "inherited from ENABLE_BACKEND_ACCESS_CONTROL",
    )
