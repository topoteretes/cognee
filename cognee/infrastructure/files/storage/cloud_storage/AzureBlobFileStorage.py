import adlfs

from cognee.infrastructure.utils.run_async import run_async
from cognee.infrastructure.files.storage.storage_provider_registry import StorageProviderRegistry
from .CloudStorageProvider import CloudStorageProvider
from .cloud_storage_config import get_cloud_storage_config


@StorageProviderRegistry.register("azure", cloud_storage_schemes=["az://"])
class AzureBlobFileStorage(CloudStorageProvider):
    """
    Manage Azure Blob file storage operations.
    Inherits common cloud storage logic from CloudStorageProvider.
    """

    @property
    def scheme(self) -> str:
        return "az://"

    def _initialize_filesystem(self) -> adlfs.AzureBlobFileSystem:
        azure_config = get_cloud_storage_config()
        if (
            azure_config.azure_storage_account_key is not None
            and azure_config.azure_storage_account_name is not None
        ):
            return adlfs.AzureBlobFileSystem(
                account_name=azure_config.azure_storage_account_name,
                account_key=azure_config.azure_storage_account_key,
            )
        else:
            raise ValueError("Azure Blob credentials are not set in the configuration.")

    async def remove(self, file_path: str):
        """
        Remove the specified file from the filesystem if it exists.
        Customized for Azure Blob Storage.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        full_file_path = self._get_full_path(file_path)

        def remove_file():
            if self.fs.exists(full_file_path) and self.fs.isfile(full_file_path):
                self.fs.rm(full_file_path, full_file_path)

        await run_async(remove_file)
