import os
from pathlib import Path
from typing import Optional
from functools import lru_cache
from cognee.root_dir import get_absolute_path, ensure_absolute_path
from cognee.modules.observability.observers import Observer
from pydantic_settings import BaseSettings, SettingsConfigDict
import pydantic


class BaseConfig(BaseSettings):
    data_root_directory: str = get_absolute_path(".data_storage")
    system_root_directory: str = get_absolute_path(".cognee_system")
    cache_root_directory: str = get_absolute_path(".cognee_cache")
    logs_root_directory: str = os.getenv(
        "COGNEE_LOGS_DIR", str(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))
    )
    monitoring_tool: object = Observer.NONE

    @pydantic.model_validator(mode="after")
    def validate_paths(self):
        # Adding this here temporarily to ensure that the cache root directory is set correctly for S3 storage automatically
        # I'll remove this after we update documentation for S3 storage
        # Auto-configure cache root directory for S3 storage if not explicitly set
        storage_backend = os.getenv("STORAGE_BACKEND", "").lower()
        cache_root_env = os.getenv("CACHE_ROOT_DIRECTORY")

        if storage_backend == "s3" and not cache_root_env:
            # Auto-generate S3 cache path when using S3 storage
            bucket_name = os.getenv("STORAGE_BUCKET_NAME")
            if bucket_name:
                self.cache_root_directory = f"s3://{bucket_name}/cognee/cache"

        # Require absolute paths for root directories
        self.data_root_directory = ensure_absolute_path(self.data_root_directory)
        self.system_root_directory = ensure_absolute_path(self.system_root_directory)
        self.logs_root_directory = ensure_absolute_path(self.logs_root_directory)

        # Set monitoring tool based on available keys
        if self.langfuse_public_key and self.langfuse_secret_key:
            self.monitoring_tool = Observer.LANGFUSE

        return self

    langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: Optional[str] = os.getenv("LANGFUSE_HOST")
    default_user_email: Optional[str] = os.getenv("DEFAULT_USER_EMAIL")
    default_user_password: Optional[str] = os.getenv("DEFAULT_USER_PASSWORD")
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "data_root_directory": self.data_root_directory,
            "system_root_directory": self.system_root_directory,
            "monitoring_tool": self.monitoring_tool,
            "cache_root_directory": self.cache_root_directory,
            "logs_root_directory": self.logs_root_directory,
        }


@lru_cache
def get_base_config():
    return BaseConfig()
