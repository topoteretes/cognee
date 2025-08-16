from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudStorageConfig(BaseSettings):
    """
    Unified Configuration for Storage
    """

    storage_backend: str = "local"  # local, s3, azure, gcs

    # AWS S3 Specific Config
    aws_region: Optional[str] = None
    aws_endpoint_url: Optional[str] = None  # TODO: check if this is needed
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Azure Blob Storage Specific Config (for future sprints)
    azure_account_name: Optional[str] = None
    azure_account_key: Optional[str] = None
    azure_container_name: Optional[str] = None

    # Google Cloud Storage Specific Config (for future sprints)
    gcs_project_id: Optional[str] = None
    gcs_credentials_path: Optional[str] = None
    gcs_bucket: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_cloud_storage_config():
    return CloudStorageConfig()
