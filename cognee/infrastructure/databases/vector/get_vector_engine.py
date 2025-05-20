from .config import get_vectordb_config
from .create_vector_engine import create_vector_engine
from cognee.context_global_variables import vector_db_config


def get_vector_engine():
    # TODO: Create new get_vector_db_context_config function to handle context variables
    if vector_db_config.get():
        return create_vector_engine(**vector_db_config.get())
    return create_vector_engine(**get_vectordb_config().to_dict())
