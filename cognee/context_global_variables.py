import os
from contextvars import ContextVar
from typing import Union
from uuid import UUID

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.utils import get_or_create_dataset_database
from cognee.infrastructure.databases.utils import resolve_dataset_database_connection_info
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.modules.users.methods import get_user

# Note: ContextVar allows us to use different graph db configurations in Cognee
#       for different async tasks, threads and processes
vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)
session_user = ContextVar("session_user", default=None)


async def set_session_user_context_variable(user):
    session_user.set(user)


def multi_user_support_possible():
    graph_db_config = get_graph_config()
    vector_db_config = get_vectordb_config()

    graph_handler = graph_db_config.graph_dataset_database_handler
    vector_handler = vector_db_config.vector_dataset_database_handler
    from cognee.infrastructure.databases.dataset_database_handler import (
        supported_dataset_database_handlers,
    )

    if graph_handler not in supported_dataset_database_handlers:
        raise EnvironmentError(
            "Unsupported graph dataset to database handler configured. Cannot add support for multi-user access control mode. Please use a supported graph dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected graph dataset to database handler: {graph_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    if vector_handler not in supported_dataset_database_handlers:
        raise EnvironmentError(
            "Unsupported vector dataset to database handler configured. Cannot add support for multi-user access control mode. Please use a supported vector dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected vector dataset to database handler: {vector_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    if (
        supported_dataset_database_handlers[graph_handler]["handler_provider"]
        != graph_db_config.graph_database_provider
    ):
        raise EnvironmentError(
            "The selected graph dataset to database handler does not work with the configured graph database provider. Cannot add support for multi-user access control mode. Please use a supported graph dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected graph database provider: {graph_db_config.graph_database_provider}\n"
            f"Selected graph dataset to database handler: {graph_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    if (
        supported_dataset_database_handlers[vector_handler]["handler_provider"]
        != vector_db_config.vector_db_provider
    ):
        raise EnvironmentError(
            "The selected vector dataset to database handler does not work with the configured vector database provider. Cannot add support for multi-user access control mode. Please use a supported vector dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected vector database provider: {vector_db_config.vector_db_provider}\n"
            f"Selected vector dataset to database handler: {vector_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    return True


def backend_access_control_enabled():
    backend_access_control = os.environ.get("ENABLE_BACKEND_ACCESS_CONTROL", None)
    if backend_access_control is None:
        # If backend access control is not defined in environment variables,
        # enable it by default if graph and vector DBs can support it, otherwise disable it
        return multi_user_support_possible()
    elif backend_access_control.lower() == "true":
        # If enabled, ensure that the current graph and vector DBs can support it
        return multi_user_support_possible()
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
    # Ensure that all connection info is resolved properly
    dataset_database = await resolve_dataset_database_connection_info(dataset_database)

    base_config = get_base_config()
    data_root_directory = os.path.join(
        base_config.data_root_directory, str(user.tenant_id or user.id)
    )
    databases_directory_path = os.path.join(
        base_config.system_root_directory, "databases", str(user.id)
    )

    # Set vector and graph database configuration based on dataset database information
    # TODO: Add better handling of vector and graph config across Cognee.
    #  LRU_CACHE takes into account order of inputs, if order of inputs is changed it will be registered as a new DB adapter
    vector_config = {
        "vector_db_provider": dataset_database.vector_database_provider,
        "vector_db_url": dataset_database.vector_database_url,
        "vector_db_key": dataset_database.vector_database_key,
        "vector_db_name": dataset_database.vector_database_name,
        "vector_db_port": dataset_database.vector_database_connection_info.get("port", ""),
        "vector_db_host": dataset_database.vector_database_connection_info.get("host", ""),
        "vector_db_username": dataset_database.vector_database_connection_info.get("username", ""),
        "vector_db_password": dataset_database.vector_database_connection_info.get("password", ""),
    }

    graph_config = {
        "graph_database_provider": dataset_database.graph_database_provider,
        "graph_database_url": dataset_database.graph_database_url,
        "graph_database_name": dataset_database.graph_database_name,
        "graph_database_key": dataset_database.graph_database_key,
        "graph_file_path": os.path.join(
            databases_directory_path, dataset_database.graph_database_name
        ),
        "graph_database_username": dataset_database.graph_database_connection_info.get(
            "graph_database_username", ""
        ),
        "graph_database_password": dataset_database.graph_database_connection_info.get(
            "graph_database_password", ""
        ),
        "graph_dataset_database_handler": "",
        "graph_database_port": "",
    }

    storage_config = {
        "data_root_directory": data_root_directory,
    }

    # Use ContextVar to use these graph and vector configurations are used
    # in the current async context across Cognee
    graph_db_config.set(graph_config)
    vector_db_config.set(vector_config)
    file_storage_config.set(storage_config)
