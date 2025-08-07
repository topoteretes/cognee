from functools import lru_cache
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path


class LoaderConfig(BaseSettings):
    """
    Configuration for file loader system.

    Follows cognee's pattern using pydantic_settings.BaseSettings for
    environment variable support and validation.
    """

    loader_directories: List[str] = [
        get_absolute_path("cognee/infrastructure/loaders/core"),
        get_absolute_path("cognee/infrastructure/loaders/external"),
    ]
    default_loader_priority: List[str] = [
        "text_loader",
        "pypdf_loader",
        "unstructured_loader",
        "dlt_loader",
    ]
    auto_discover: bool = True
    fallback_loader: str = "text_loader"
    enable_dependency_validation: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow", env_prefix="LOADER_")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dict containing all loader configuration settings
        """
        return {
            "loader_directories": self.loader_directories,
            "default_loader_priority": self.default_loader_priority,
            "auto_discover": self.auto_discover,
            "fallback_loader": self.fallback_loader,
            "enable_dependency_validation": self.enable_dependency_validation,
        }


@lru_cache
def get_loader_config() -> LoaderConfig:
    """
    Get cached loader configuration.

    Uses LRU cache following cognee's pattern for configuration objects.

    Returns:
        LoaderConfig instance with current settings
    """
    return LoaderConfig()
