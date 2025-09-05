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
    aws_endpoint_url: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Azure Blob Storage Specific Config
    azure_storage_account_name: Optional[str] = None
    azure_storage_account_key: Optional[str] = None
    azure_storage_container_name: Optional[str] = None

    # Google Cloud Storage Specific Config
    google_project_id: Optional[str] = None
    google_application_credentials: Optional[str] = None
    google_bucket: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        """
        Return the configuration as a dictionary.

        This dictionary contains all the configurations related to the cloud storage.

        Returns:
        --------

            - dict: A dictionary representation of the configuration settings.
        """
        return {
            "storage_backend": self.storage_backend,
            "aws_region": self.aws_region,
            "aws_endpoint_url": self.aws_endpoint_url,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "azure_storage_account_name": self.azure_storage_account_name,
            "azure_storage_account_key": self.azure_storage_account_key,
            "azure_storage_container_name": self.azure_storage_container_name,
            "google_project_id": self.google_project_id,
            "google_application_credentials": self.google_application_credentials,
            "google_bucket": self.google_bucket,
            "model_config": self.model_config,
        }

    def to_hashable_dict(self) -> dict:
        """
        Return a hashable dictionary with essential cloud storage configuration parameters.

        This dictionary excludes certain non-hashable objects and focuses on unique identifiers
        for cloud storage configurations.

        Returns:
        --------

            - dict: A dictionary representation of the essential cloud storage configuration
                settings.
        """
        return {
            "storage_backend": self.storage_backend,
            "aws_region": self.aws_region,
            "aws_endpoint_url": self.aws_endpoint_url,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "azure_storage_account_name": self.azure_storage_account_name,
            "azure_storage_account_key": self.azure_storage_account_key,
            "azure_storage_container_name": self.azure_storage_container_name,
            "google_project_id": self.google_project_id,
            "google_application_credentials": self.google_application_credentials,
            "google_bucket": self.google_bucket,
        }


@lru_cache
def get_cloud_storage_config():
    """
    Retrieve the cloud storage configuration. This function utilizes caching to return a singleton
    instance of the CloudStorageConfig class for efficiency.

    It creates and returns a CloudStorageConfig object, which contains various settings related to
    cloud storage configuration.

    Returns:
    --------

        - CloudStorageConfig: A CloudStorageConfig instance containing the cloud storage configuration settings.
    """
    return CloudStorageConfig()
