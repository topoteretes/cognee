from .api.v1.add import add
from .api.v1.cognify import cognify
from .api.v1.config.config import config
from .api.v1.datasets.datasets import datasets
from .api.v1.prune import prune
from .api.v1.search import SearchType, get_search_history, search
# Pipelines
from .modules import pipelines

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass
