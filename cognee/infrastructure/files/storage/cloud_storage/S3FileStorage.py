import s3fs

from cognee.infrastructure.files.storage.storage_provider_registry import StorageProviderRegistry
from .CloudStorageProvider import CloudStorageProvider
from .cloud_storage_config import get_cloud_storage_config


@StorageProviderRegistry.register("s3", cloud_storage_schemes=["s3://"])
class S3FileStorage(CloudStorageProvider):
    """
    Manage S3 file storage operations.
    Inherits common cloud storage logic from CloudStorageProvider.
    """

    @property
    def scheme(self) -> str:
        return "s3://"

    def _initialize_filesystem(self) -> s3fs.S3FileSystem:
        s3_config = get_cloud_storage_config()
        if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
            return s3fs.S3FileSystem(
                key=s3_config.aws_access_key_id,
                secret=s3_config.aws_secret_access_key,
                anon=False,
                endpoint_url=s3_config.aws_endpoint_url,
                client_kwargs={"region_name": s3_config.aws_region},
            )
        else:
            raise ValueError("S3 credentials are not set in the configuration.")
