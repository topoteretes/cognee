import os
from cognee.infrastructure.files.storage import StorageProviderRegistry
from cognee.infrastructure.files.storage.utils import get_scheme_with_separator
from cognee.infrastructure.databases.mixins.cloud_database_mixin import CloudDatabaseMixin
from cognee.shared.logging_utils import get_logger


logger = get_logger()


class KuzuCloudDatabaseMixin(CloudDatabaseMixin):
    """
    Provides functionality for synchronizing a local Kuzu database file
    with a cloud storage backend (e.g., S3).

    This mixin expects the consuming class to provide the following attributes:
    - self.db_path (str): The cloud storage URI for the database.
    - self.temp_graph_file (str): The local temporary path for the database directory.
    - self.connection (kuzu.Connection): The active Kuzu connection or None.
    - self.KUZU_ASYNC_LOCK (asyncio.Lock): An asyncio lock for concurrency control.
    """

    async def push_to_cloud(self) -> None:
        """
        Pushes the local temporary database file to cloud storage.
        """
        if os.getenv("STORAGE_BACKEND", "local").lower() != "local" and hasattr(
            self, "temp_graph_file"
        ):
            scheme_with_separator = get_scheme_with_separator(self.db_path)
            cloud_storage_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
                scheme_with_separator
            )
            cloud_storage = cloud_storage_cls("")

            if self.connection:
                async with self.KUZU_ASYNC_LOCK:
                    self.connection.execute("CHECKPOINT;")

            cloud_storage.fs.put(self.temp_graph_file, self.db_path, recursive=True)

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
            cloud_storage.fs.get(self.db_path, self.temp_graph_file, recursive=True)
        except FileNotFoundError:
            logger.warning(f"Kuzu cloud storage file not found: {self.db_path}")
