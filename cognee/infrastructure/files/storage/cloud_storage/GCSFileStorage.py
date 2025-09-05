import gcsfs

from cognee.infrastructure.files.storage.storage_provider_registry import StorageProviderRegistry
from .CloudStorageProvider import CloudStorageProvider
from .cloud_storage_config import get_cloud_storage_config


@StorageProviderRegistry.register("gcs", cloud_storage_schemes=["gs://"])
class GCSFileStorage(CloudStorageProvider):
    """
    Manage GCS file storage operations.
    Inherits common cloud storage logic from CloudStorageProvider.
    """

    @property
    def scheme(self) -> str:
        return "gs://"

    def _initialize_filesystem(self) -> gcsfs.GCSFileSystem:
        gcs_config = get_cloud_storage_config()
        if (
            gcs_config.google_project_id is not None
            and gcs_config.google_application_credentials is not None
        ):
            return gcsfs.GCSFileSystem(
                project=gcs_config.google_project_id,
                token=gcs_config.google_application_credentials,
            )
        else:
            raise ValueError("GCS credentials are not set in the configuration.")
