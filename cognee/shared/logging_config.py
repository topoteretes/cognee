from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    log_level: str = "INFO"
    cognee_log_file: bool = True
    cognee_logs_dir: Optional[str] = None
    cognee_log_max_bytes: int = 50 * 1024 * 1024
    cognee_log_backup_count: int = 5
    cognee_log_search_history: bool = True
    litellm_log: Optional[str] = None
    cognee_cli_mode: bool = False
    cognee_minimal_logging: bool = False
    log_file_name: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "log_level": self.log_level,
            "cognee_log_file": self.cognee_log_file,
            "cognee_logs_dir": self.cognee_logs_dir,
            "cognee_log_max_bytes": self.cognee_log_max_bytes,
            "cognee_log_backup_count": self.cognee_log_backup_count,
            "cognee_log_search_history": self.cognee_log_search_history,
            "litellm_log": self.litellm_log,
            "cognee_cli_mode": self.cognee_cli_mode,
            "cognee_minimal_logging": self.cognee_minimal_logging,
            "log_file_name": self.log_file_name,
        }


@lru_cache
def get_logging_config():
    return LoggingConfig()
