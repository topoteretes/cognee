import os
from contextvars import ContextVar
from typing import Union
from uuid import UUID

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.infrastructure.databases.utils import get_or_create_dataset_database
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.modules.users.methods import get_user

# Note: ContextVar allows us to use different graph db configurations in Cognee
#       for different async tasks, threads and processes
vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)
session_user = ContextVar("session_user", default=None)

VECTOR_DBS_WITH_MULTI_USER_SUPPORT = ["lancedb"]
GRAPH_DBS_WITH_MULTI_USER_SUPPORT = ["kuzu"]


async def set_session_user_context_variable(user):
    session_user.set(user)


def multi_user_support_possible():
    graph_db_config = get_graph_context_config()
    vector_db_config = get_vectordb_context_config()
    return (
        graph_db_config["graph_database_provider"] in GRAPH_DBS_WITH_MULTI_USER_SUPPORT
        and vector_db_config["vector_db_provider"] in VECTOR_DBS_WITH_MULTI_USER_SUPPORT
    )


def backend_access_control_enabled():
    backend_access_control = os.environ.get("ENABLE_BACKEND_ACCESS_CONTROL", None)
    if backend_access_control is None:
        # If backend access control is not defined in environment variables,
        # enable it by default if graph and vector DBs can support it, otherwise disable it
        return multi_user_support_possible()
    elif backend_access_control.lower() == "true":
        # If enabled, ensure that the current graph and vector DBs can support it
        multi_user_support = multi_user_support_possible()
        if not multi_user_support:
            raise EnvironmentError(
                "ENABLE_BACKEND_ACCESS_CONTROL is set to true but the current graph and/or vector databases do not support multi-user access control. Please use supported databases or disable backend access control."
            )
        return True
    return False


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

    if not backend_access_control_enabled():
        return

    user = await get_user(user_id)

    # To ensure permissions are enforced properly all datasets will have their own databases
    dataset_database = await get_or_create_dataset_database(dataset, user)

    base_config = get_base_config()
    data_root_directory = os.path.join(
        base_config.data_root_directory, str(user.tenant_id or user.id)
    )
    databases_directory_path = os.path.join(
        base_config.system_root_directory, "databases", str(user.id)
    )

    # Set vector and graph database configuration based on dataset database information
    vector_config = {
        "vector_db_provider": dataset_database.vector_database_provider,
        "vector_db_url": dataset_database.vector_database_url,
        "vector_db_key": dataset_database.vector_database_key,
        "vector_db_name": dataset_database.vector_database_name,
    }

    graph_config = {
        "graph_database_provider": dataset_database.graph_database_provider,
        "graph_database_url": dataset_database.graph_database_url,
        "graph_database_name": dataset_database.graph_database_name,
        "graph_database_key": dataset_database.graph_database_key,
        "graph_file_path": os.path.join(
            databases_directory_path, dataset_database.graph_database_name
        ),
    }

    storage_config = {
        "data_root_directory": data_root_directory,
    }

    # Use ContextVar to use these graph and vector configurations are used
    # in the current async context across Cognee
    graph_db_config.set(graph_config)
    vector_db_config.set(vector_config)
    file_storage_config.set(storage_config)
