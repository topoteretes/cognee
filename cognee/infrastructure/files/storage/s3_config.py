from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Config(BaseSettings):
    aws_region: Optional[str] = None
    aws_endpoint_url: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_s3_config():
    return S3Config()
