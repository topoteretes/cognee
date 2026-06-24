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
    logs_root_directory: str = os.getenv("COGNEE_LOGS_DIR", str(Path.home() / ".cognee" / "logs"))
    monitoring_tool: object = Observer.NONE
    # Default blend weight for the learned feedback signal during graph search.
    # Opt-in by default to preserve existing retrieval behavior.
    default_feedback_influence: float = float(os.getenv("DEFAULT_FEEDBACK_INFLUENCE", "0.0"))

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

        return self

    default_user_email: Optional[str] = os.getenv("DEFAULT_USER_EMAIL")
    default_user_password: Optional[str] = os.getenv("DEFAULT_USER_PASSWORD")

    # OpenTelemetry / tracing
    cognee_tracing_enabled: bool = os.getenv("COGNEE_TRACING_ENABLED", "false")
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "cognee")
    otel_exporter_otlp_endpoint: Optional[str] = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_exporter_otlp_headers: Optional[str] = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

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
