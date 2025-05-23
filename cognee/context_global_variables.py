import os
import pathlib
from contextvars import ContextVar
from typing import Union
from uuid import UUID

from cognee.infrastructure.databases.utils import get_or_create_dataset_database
from cognee.modules.users.methods import get_user

# Note: ContextVar allows us to use different graph db configurations in Cognee
#       for different async tasks, threads and processes
vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)


async def set_database_global_context_variables(dataset: Union[str, UUID], user_id: UUID):
    """
    If backend access control is enabled this function will ensure all datasets have their own databases,
    access to which will be enforced by given permissions.
    Database name will be determined by dataset_id and LanceDB and KuzuDB use will be enforced.

    Note: This is only currently supported by the following databases:
          Relational: SQLite, Postgres
          Vector: LanceDB
          Graph: KuzuDB

    Args:
        dataset: Cognee dataset name or id
        user_id: UUID of the owner of the dataset

    Returns:

    """

    if not os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true":
        return

    user = await get_user(user_id)

    # To ensure permissions are enforced properly all datasets will have their own databases
    dataset_database = await get_or_create_dataset_database(dataset, user)

    # TODO: Find better location for database files
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, f".cognee_system/databases/{user.id}")
        ).resolve()
    )

    # Set vector and graph database configuration based on dataset database information
    vector_config = {
        "vector_db_url": os.path.join(cognee_directory_path, dataset_database.vector_database_name),
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
    }

    graph_config = {
        "graph_database_provider": "kuzu",
        "graph_file_path": os.path.join(
            cognee_directory_path, dataset_database.graph_database_name
        ),
    }

    # Use ContextVar to use these graph and vector configurations are used
    # in the current async context across Cognee
    graph_db_config.set(graph_config)
    vector_db_config.set(vector_config)
