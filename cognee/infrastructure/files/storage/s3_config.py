from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Config(BaseSettings):
    aws_region: str | None = None
    aws_endpoint_url: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_profile_name: str | None = None
    aws_bedrock_runtime_endpoint: str | None = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_s3_config():
    return S3Config()
