from cognee.infrastructure.loaders.external import PyPdfLoader
from cognee.infrastructure.loaders.core import TextLoader, AudioLoader, ImageLoader

# Registry for loader implementations
supported_loaders = {
    PyPdfLoader.loader_name: PyPdfLoader,
    TextLoader.loader_name: TextLoader,
    ImageLoader.loader_name: ImageLoader,
    AudioLoader.loader_name: AudioLoader,
}

# Try adding optional loaders
try:
    from cognee.infrastructure.loaders.external import UnstructuredLoader

    supported_loaders[UnstructuredLoader.loader_name] = UnstructuredLoader
except ImportError:
    pass
