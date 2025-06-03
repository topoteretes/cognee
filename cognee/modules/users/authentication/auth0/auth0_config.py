from functools import lru_cache
from pydantic_settings import BaseSettings


class Auth0Config(BaseSettings):
    auth0_domain: str
    auth0_api_audience: str
    auth0_algorithms: str
    auth0_client_secret: str
    auth0_client_id: str
    auth0_issuer: str
    auth_token_cookie_name: str

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_auth0_config():
    return Auth0Config()
