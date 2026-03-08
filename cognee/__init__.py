# ruff: noqa: E402
from typing import Any, Optional
from uuid import UUID

from cognee.version import get_cognee_version

# NOTE: __version__ extraction must be at the top of the __init__.py otherwise
#       there will be circular import issues
__version__ = get_cognee_version()

# Load environment variable settings has to be before setting up logging for LOG_LEVEL value
import dotenv

dotenv.load_dotenv(override=True)

# NOTE: Log level can be set with the LOG_LEVEL env variable
from cognee.shared.logging_utils import setup_logging

logger = setup_logging()

from .api.v1.add import add as _add
from .api.v1.delete import delete as _delete
from .api.v1.cognify import cognify as _cognify
from .modules.memify import memify as _memify
from .modules.run_custom_pipeline import run_custom_pipeline
from .api.v1.update import update
from .api.v1.config.config import config as _config
from .api.v1.datasets.datasets import datasets as _datasets
from .api.v1.prune import prune
from .api.v1.search import SearchType, search as _search
from .api.v1.visualize import visualize_graph, start_visualization_server
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from .api.v1.ui import start_ui
from .api.v1.session import session

# Pipelines
from .modules import pipelines

from cognee.run_migrations import run_migrations

# Tracing / Observability
from cognee.modules.observability.trace_context import (
    enable_tracing,
    disable_tracing,
    get_last_trace,
    get_all_traces,
    clear_traces,
)


async def add(*args: Any, **kwargs: Any):
    """Ingest data into a Cognee dataset.

    Args:
        *args: Positional arguments forwarded to :func:`cognee.api.v1.add.add`.
        **kwargs: Keyword arguments forwarded to :func:`cognee.api.v1.add.add`
            (for example ``data``, ``dataset_name``, ``user``, ``node_set``).

    Returns:
        Pipeline run metadata from the ingestion pipeline.

    Example:
        >>> import cognee
        >>> await cognee.add("Cognee builds AI memory.", dataset_name="demo")
    """

    return await _add(*args, **kwargs)


async def cognify(*args: Any, **kwargs: Any):
    """Process ingested data into knowledge graph structures.

    Args:
        *args: Positional arguments forwarded to :func:`cognee.api.v1.cognify.cognify`.
        **kwargs: Keyword arguments for cognify execution (for example ``datasets``,
            ``graph_model``, ``chunk_size``, ``run_in_background``).

    Returns:
        Pipeline run metadata/result for the cognify pipeline.

    Example:
        >>> import cognee
        >>> await cognee.cognify(datasets="demo")
    """

    return await _cognify(*args, **kwargs)


async def search(*args: Any, **kwargs: Any):
    """Run semantic or graph-aware search on previously cognified data.

    Args:
        *args: Positional arguments forwarded to :func:`cognee.api.v1.search.search`.
        **kwargs: Keyword arguments for search (for example ``query_text``,
            ``query_type``, ``datasets``, ``top_k``).

    Returns:
        A list of search results (shape depends on ``query_type``).

    Example:
        >>> import cognee
        >>> await cognee.search("What did we ingest?", query_type=cognee.SearchType.GRAPH_COMPLETION)
    """

    return await _search(*args, **kwargs)


async def memify(*args: Any, **kwargs: Any):
    """Run the memify enrichment pipeline on graph/data context.

    Args:
        *args: Positional arguments forwarded to :func:`cognee.modules.memify.memify`.
        **kwargs: Keyword arguments for memify execution (for example ``dataset``,
            ``extraction_tasks``, ``enrichment_tasks``, ``run_in_background``).

    Returns:
        Pipeline run metadata/result for memify.

    Example:
        >>> import cognee
        >>> await cognee.memify(dataset="demo")
    """

    return await _memify(*args, **kwargs)


async def delete(data_id: UUID, dataset_id: UUID, mode: str = "soft", user: Optional[Any] = None):
    """Delete a dataset item by id (deprecated compatibility API).

    Args:
        data_id: Identifier of the data record to remove.
        dataset_id: Identifier of the dataset that contains the data.
        mode: Deletion mode. Kept for backward compatibility.
        user: Optional user context for authorization checks.

    Returns:
        Result of the underlying delete operation.

    Example:
        >>> import cognee
        >>> await cognee.delete(data_id=item_id, dataset_id=dataset_id)
    """

    return await _delete(data_id=data_id, dataset_id=dataset_id, mode=mode, user=user)


def config():
    """Return the global configuration helper class.

    Args:
        None.

    Returns:
        The ``cognee.api.v1.config.config`` class with static setters.

    Example:
        >>> import cognee
        >>> cognee.config().set_llm_provider("openai")
    """

    return _config


def datasets():
    """Return the datasets helper class for dataset/list/delete utilities.

    Args:
        None.

    Returns:
        The ``cognee.api.v1.datasets.datasets`` class.

    Example:
        >>> import cognee
        >>> all_datasets = await cognee.datasets().list_datasets()
    """

    return _datasets
