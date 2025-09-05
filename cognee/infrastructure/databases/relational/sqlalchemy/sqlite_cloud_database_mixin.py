import os
from cognee.infrastructure.databases.mixins.cloud_database_mixin import CloudDatabaseMixin
from cognee.infrastructure.files.storage import StorageProviderRegistry
from cognee.infrastructure.files.storage.utils import get_scheme_with_separator


class SQLiteCloudDatabaseMixin(CloudDatabaseMixin):
    """
    Provides functionality for synchronizing a local SQLite database file
    with a cloud storage backend (e.g., S3).
    """

    async def push_to_cloud(self) -> None:
        """
        Pushes the local temporary database file to cloud storage.
        """
        if os.getenv("STORAGE_BACKEND", "local").lower() != "local" and hasattr(
            self, "temp_db_file"
        ):
            scheme_with_separator = get_scheme_with_separator(self.db_path)
            cloud_storage_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
                scheme_with_separator
            )
            cloud_storage = cloud_storage_cls("")
            cloud_storage.fs.put(self.temp_db_file, self.db_path, recursive=True)

    async def pull_from_cloud(self) -> None:
        """
        Pulls the database file from cloud storage to the local temporary file.
        """
        scheme_with_separator = get_scheme_with_separator(self.db_path)
        cloud_storage_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
            scheme_with_separator
        )
        cloud_storage = cloud_storage_cls("")

        try:
            cloud_storage.fs.get(self.db_path, self.temp_db_file, recursive=True)
        except FileNotFoundError:
            # It's okay if the file doesn't exist yet on cloud storage; it will be created.
            pass
