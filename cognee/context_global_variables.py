from contextvars import ContextVar
from typing import Union
from uuid import UUID

from cognee.api.v1.infrastructure import get_or_create_dataset_database
from cognee.modules.users.models import User

# Note: ContextVar allows us to use different graph db configurations in Cognee
#       for different async tasks, threads and processes
vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)


async def set_database_global_context_variables(dataset: Union[str, UUID], user: User):
    # TODO: Add use of context databases to be optional depending on the need for permissions
    #      as they don't allow the use of other databases currently

    # To ensure permissions are enforced properly all datasets will have their own databases
    dataset_database = await get_or_create_dataset_database(dataset, user)

    # Set vector and graph database configuration based on dataset database information
    vector_config = {
        "vector_db_url": dataset_database.vector_database_name,
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
    }

    graph_config = {
        "graph_database_provider": "kuzu",
        "graph_file_path": dataset_database.graph_database_name,
    }

    # Use ContextVar to use these graph and vector configurations are used
    # in the current async context across Cognee
    graph_db_config.set(graph_config)
    vector_db_config.set(vector_config)
