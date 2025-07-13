from .supported_loaders import supported_loaders


def use_loader(loader_name: str, loader_class):
    """
    Register a loader at runtime.

    Follows cognee's pattern used in databases for adapter registration.
    This allows external packages and custom loaders to be registered
    into the loader system.

    Args:
        loader_name: Unique name for the loader
        loader_class: Loader class implementing LoaderInterface

    Example:
        from cognee.infrastructure.loaders import use_loader
        from my_package import MyCustomLoader

        use_loader("my_custom_loader", MyCustomLoader)
    """
    supported_loaders[loader_name] = loader_class
