# from functools import lru_cache

from .config import get_relational_config
from .create_relational_engine import create_relational_engine

# @lru_cache
def get_relational_engine():
    relational_config = get_relational_config()

    return create_relational_engine(**relational_config.to_dict())