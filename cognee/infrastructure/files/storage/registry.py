from typing import Type, Dict, Callable
from .storage_provider import CloudStorageProvider


class StorageProviderRegistry:
    """
    A registry for dynamically managing and retrieving
    all available CloudStorageProvider implementations.
    """

    _providers: Dict[str, Type[CloudStorageProvider]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Type[CloudStorageProvider]], Type[CloudStorageProvider]]:
        """
        A class decorator for registering storage providers to the system.

        Usage:
        @StorageProviderRegistry.register("s3")
        class S3StorageProvider(CloudStorageProvider):
        ...
        """

        def decorator(provider_class: Type[CloudStorageProvider]) -> Type[CloudStorageProvider]:
            if name in cls._providers:
                raise ValueError(f"Provider with name '{name}' is already registered.")
            cls._providers[name] = provider_class
            return provider_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[CloudStorageProvider]:
        """
        Get a storage provider by name.

        Parameters:
        -----------

            - name (str): The name of the storage provider to get.

        Returns:
        --------

            - Type[CloudStorageProvider]: The storage provider class.

        Raises:
        -------

            - ValueError: If the storage provider is not found.
        """
        try:
            return cls._providers[name]
        except KeyError as exc:
            raise ValueError(
                f"No storage provider registered with the name '{name}'. "
                f"Available providers: {list(cls._providers.keys())}"
            ) from exc
