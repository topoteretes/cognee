from cognee.infrastructure.loaders import LoaderInterface
from cognee.infrastructure.loaders.core import (
    AudioLoader,
    CodeLoader,
    CsvLoader,
    ImageLoader,
    TextLoader,
    VideoLoader,
)
from cognee.infrastructure.loaders.external import PyPdfLoader

# Registry for loader implementations
supported_loaders: dict[str, type[LoaderInterface]] = {
    PyPdfLoader.loader_name: PyPdfLoader,
    TextLoader.loader_name: TextLoader,
    CodeLoader.loader_name: CodeLoader,
    ImageLoader.loader_name: ImageLoader,
    AudioLoader.loader_name: AudioLoader,
    VideoLoader.loader_name: VideoLoader,
    CsvLoader.loader_name: CsvLoader,
}

# Try adding optional loaders
try:
    from cognee.infrastructure.loaders.external import UnstructuredLoader

    supported_loaders[UnstructuredLoader.loader_name] = UnstructuredLoader
except ImportError:
    pass

try:
    from cognee.infrastructure.loaders.external import AdvancedPdfLoader

    supported_loaders[AdvancedPdfLoader.loader_name] = AdvancedPdfLoader
except ImportError:
    pass

try:
    from cognee.infrastructure.loaders.external import BeautifulSoupLoader

    supported_loaders[BeautifulSoupLoader.loader_name] = BeautifulSoupLoader
except ImportError:
    pass

try:
    from cognee.infrastructure.loaders.external import DoclingLoader

    supported_loaders[DoclingLoader.loader_name] = DoclingLoader
except ImportError:
    pass

# Structured CSV ingestion through dlt; registered above csv_loader in the
# engine's priority order, so with the dlt extra installed CSVs take the
# DLT manifest route instead of text flattening.
try:
    from cognee.infrastructure.loaders.external import DltCsvLoader

    supported_loaders[DltCsvLoader.loader_name] = DltCsvLoader
except ImportError:
    pass
