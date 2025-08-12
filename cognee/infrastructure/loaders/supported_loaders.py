from cognee.infrastructure.loaders.external import PyPdfLoader
from cognee.infrastructure.loaders.core import TextLoader

# Registry for loader implementations
supported_loaders = {
    PyPdfLoader.loader_name: PyPdfLoader,
    TextLoader.loader_name: TextLoader,
}
