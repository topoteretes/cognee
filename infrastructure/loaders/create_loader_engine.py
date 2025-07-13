from typing import List
from .LoaderEngine import LoaderEngine
from .supported_loaders import supported_loaders


def create_loader_engine(
    loader_directories: List[str],
    default_loader_priority: List[str],
    auto_discover: bool = True,
    fallback_loader: str = "text_loader",
    enable_dependency_validation: bool = True,
) -> LoaderEngine:
    """
    Create loader engine with given configuration.

    Follows cognee's pattern for engine creation functions used
    in database adapters.

    Args:
        loader_directories: Directories to search for loader implementations
        default_loader_priority: Priority order for loader selection
        auto_discover: Whether to auto-discover loaders from directories
        fallback_loader: Default loader to use when no other matches
        enable_dependency_validation: Whether to validate loader dependencies

    Returns:
        Configured LoaderEngine instance
    """
    engine = LoaderEngine(
        loader_directories=loader_directories,
        default_loader_priority=default_loader_priority,
        fallback_loader=fallback_loader,
        enable_dependency_validation=enable_dependency_validation,
    )

    # Register supported loaders from registry
    for loader_name, loader_class in supported_loaders.items():
        try:
            loader_instance = loader_class()
            engine.register_loader(loader_instance)
        except Exception as e:
            # Log but don't fail - allow engine to continue with other loaders
            engine.logger.warning(f"Failed to register loader {loader_name}: {e}")

    # Auto-discover loaders if enabled
    if auto_discover:
        engine.discover_loaders()

    return engine
