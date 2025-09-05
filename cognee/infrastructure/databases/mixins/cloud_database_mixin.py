import abc


class CloudDatabaseMixin(abc.ABC):
    """
    A base abstract class for cloud database mixin,
    which defines the unified interface for cloud storage synchronization.

    All concrete cloud storage mixins (e.g., for SQLite or Kuzu) must inherit this class and
    provide their own implementation for all methods marked with @abstractmethod.

    This ensures consistency and extensibility of the system.
    """

    @abc.abstractmethod
    async def push_to_cloud(self) -> None:
        """
        Push the local database file or directory to the cloud storage.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def pull_from_cloud(self) -> None:
        """
        Pull the database file or directory from the cloud storage to the local temporary location.
        """
        raise NotImplementedError
