from .LoaderEngine import LoaderEngine
from .supported_loaders import supported_loaders
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


def create_loader_engine() -> LoaderEngine:
    """
    Create loader engine with given configuration.

    Follows cognee's pattern for engine creation functions used
    in database adapters.

    Args:
        default_loader_priority: Priority order for loader selection

    Returns:
        Configured LoaderEngine instance
    """
    engine = LoaderEngine()

    # Register supported loaders from registry
    for loader_name, loader_class in supported_loaders.items():
        try:
            loader_instance = loader_class()
            engine.register_loader(loader_instance)
        except Exception as e:
            # Log but don't fail - allow engine to continue with other loaders
            logger.warning(f"Failed to register loader {loader_name}: {e}")

    return engine
