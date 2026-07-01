from functools import lru_cache
from typing import Optional, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class GoogleDriveConfig(BaseSettings):
    google_drive_folder_id: Optional[str] = None
    google_drive_auth_mode: Literal["service_account", "oauth"] = "service_account"
    google_drive_credentials_path: Optional[str] = None
    google_drive_token_path: Optional[str] = None
    google_drive_include_subfolders: bool = True
    google_drive_max_file_size_mb: int = 25

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_google_drive_config():
    return GoogleDriveConfig()
