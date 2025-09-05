from typing import List, Type, Dict, Callable, Tuple, Optional
from .storage import Storage


class StorageProviderRegistry:
    """
    A registry for dynamically managing and retrieving
    all available CloudStorageProvider implementations.
    """

    _providers: Dict[str, Type[Storage]] = {}  # storage name -> provider
    _cloud_storage_scheme_map: Dict[str, str] = {}  # cloud storage scheme -> storage name

    @classmethod
    def register(
        cls, name: str, cloud_storage_schemes: Optional[List[str]] = None
    ) -> Callable[[Type[Storage]], Type[Storage]]:
        """
        A class decorator for registering storage providers to the system.

        Usage:
        @StorageProviderRegistry.register("s3")
        class S3StorageProvider(CloudStorageProvider):
        ...
        """

        def decorator(provider_class: Type[Storage]) -> Type[Storage]:
            # register provider's name
            if name in cls._providers:
                raise ValueError(f"Provider with name '{name}' is already registered.")
            cls._providers[name] = provider_class

            if cloud_storage_schemes is not None:
                for cloud_storage_scheme in cloud_storage_schemes:
                    if cloud_storage_scheme in cls._cloud_storage_scheme_map:
                        raise ValueError(
                            f"Cloud Storage Scheme '{cloud_storage_scheme}' is already registered."
                        )
                    cls._cloud_storage_scheme_map[cloud_storage_scheme] = name

            return provider_class

        return decorator

    @classmethod
    def get_provider_by_name(cls, name: str) -> Type[Storage]:
        """
        Get a storage provider by name.

        Parameters:
        -----------

            - name (str): The name of the storage provider to get.

        Returns:
        --------

            - Type[Storage]: The storage provider class.

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

    @classmethod
    def get_provider_by_cloud_scheme(cls, cloud_storage_scheme: str) -> Type[Storage]:
        """
        Get a storage provider by scheme.
        """
        try:
            return cls._providers[cls._cloud_storage_scheme_map[cloud_storage_scheme]]
        except KeyError as exc:
            raise ValueError(
                f"No cloud storage provider registered with the scheme '{cloud_storage_scheme}'. "
                f"Available cloud storageschemes: {list(cls._cloud_storage_scheme_map.keys())}"
            ) from exc

    @classmethod
    def get_name_by_cloud_scheme(cls, cloud_storage_scheme: str) -> str:
        """
        Get a storage provider name by scheme.
        """
        try:
            return cls._cloud_storage_scheme_map[cloud_storage_scheme]
        except KeyError as exc:
            raise ValueError(
                f"No cloud storage provider registered with the scheme '{cloud_storage_scheme}'. "
                f"Available cloud storage schemes: {list(cls._cloud_storage_scheme_map.keys())}"
            ) from exc

    @classmethod
    def get_all_cloud_schemes(cls) -> Tuple[str, ...]:
        """
        Get all schemes from all registered storage providers.
        """
        return tuple(cls._cloud_storage_scheme_map.keys())
