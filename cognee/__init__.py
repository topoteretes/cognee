# ruff: noqa: E402
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

from .api.v1.add import add
from .api.v1.delete import delete
from .api.v1.cognify import cognify
from .modules.memify import memify
from .api.v1.update import update
from .api.v1.config.config import config
from .api.v1.datasets.datasets import datasets
from .api.v1.prune import prune
from .api.v1.search import SearchType, search
from .api.v1.visualize import visualize_graph, start_visualization_server
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from .api.v1.ui import start_ui

# Pipelines
from .modules import pipelines
