from contextvars import ContextVar
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseSettings):
    storage_backend: str = "local"
    storage_bucket_name: Optional[str] = None
    cache_root_directory: Optional[str] = None
    aws_region: Optional[str] = None
    aws_endpoint_url: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    aws_profile_name: Optional[str] = None
    aws_bedrock_runtime_endpoint: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "storage_backend": self.storage_backend,
            "storage_bucket_name": self.storage_bucket_name,
            "cache_root_directory": self.cache_root_directory,
            "aws_region": self.aws_region,
            "aws_endpoint_url": self.aws_endpoint_url,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
            "aws_profile_name": self.aws_profile_name,
            "aws_bedrock_runtime_endpoint": self.aws_bedrock_runtime_endpoint,
        }


@lru_cache
def get_storage_config():
    return StorageConfig()


file_storage_config = ContextVar("file_storage_config", default=None)
