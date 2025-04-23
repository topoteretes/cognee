from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Config(BaseSettings):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_s3_config():
    return S3Config()
