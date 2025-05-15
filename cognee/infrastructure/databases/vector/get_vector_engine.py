from .config import get_vectordb_config
from .create_vector_engine import create_vector_engine
from contextvars import ContextVar


# Think of this as a “global” task-local
vector_db_config = ContextVar("vector_db_config", default=None)


def get_vector_engine():
    if vector_db_config.get():
        return create_vector_engine(**vector_db_config.get())
    return create_vector_engine(**get_vectordb_config().to_dict())
