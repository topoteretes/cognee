import os
from contextvars import ContextVar
from typing import Union
from uuid import UUID

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.utils import get_or_create_dataset_database
from cognee.infrastructure.databases.utils.constants import (
    DEFAULT_GRAPH_DB_PROVIDER,
    GRAPH_DBS_WITH_MULTI_USER_SUPPORT,
)
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.modules.users.methods import get_user
from cognee.modules.data.methods import get_dataset

# Note: ContextVar allows us to use different graph db configurations in Cognee
#       for different async tasks, threads and processes
vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)
session_user = ContextVar("session_user", default=None)


async def set_session_user_context_variable(user):
    session_user.set(user)


async def set_database_global_context_variables(dataset: Union[str, UUID], user_id: UUID):
    """
    If backend access control is enabled this function will ensure all datasets have their own databases,
    access to which will be enforced by given permissions.
    Database name will be determined by dataset_id and LanceDB and the configured graph provider
    will be enforced.

    Note: This is only currently supported by the following databases:
          Relational: SQLite, Postgres
          Vector: LanceDB
          Graph: KuzuDB (default), Ladybug

    Args:
        dataset: Cognee dataset name or id
        user_id: UUID of the owner of the dataset

    Returns:

    """

    base_config = get_base_config()

    if not os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true":
        return

    user = await get_user(user_id)

    configured_graph_provider = get_graph_config().graph_database_provider
    graph_provider = (configured_graph_provider or DEFAULT_GRAPH_DB_PROVIDER).lower()
    if graph_provider not in GRAPH_DBS_WITH_MULTI_USER_SUPPORT:
        graph_provider = DEFAULT_GRAPH_DB_PROVIDER

    # To ensure permissions are enforced properly all datasets will have their own databases
    dataset_database = await get_or_create_dataset_database(
        dataset, user, graph_provider=graph_provider
    )

    dataset_name: str | None = None
    if isinstance(dataset, str):
        dataset_name = dataset
    else:
        dataset_obj = await get_dataset(user.id, dataset)
        if dataset_obj:
            dataset_name = dataset_obj.name

    if not dataset_name:
        dataset_name = str(dataset)
    project_segment = dataset_name.split("_", 1)[0]
    if not project_segment:
        project_segment = str(user.tenant_id or user.id)

    project_root = os.path.join(base_config.system_root_directory, project_segment)
    graph_root = os.path.join(project_root, "graph")
    vector_root = os.path.join(project_root, "vector")
    files_root = os.path.join(base_config.data_root_directory, project_segment, "files")

    if not graph_root.startswith("s3://"):
        os.makedirs(graph_root, exist_ok=True)
    if not vector_root.startswith("s3://"):
        os.makedirs(vector_root, exist_ok=True)
    if not files_root.startswith("s3://"):
        os.makedirs(files_root, exist_ok=True)

    # Set vector and graph database configuration based on dataset database information
    vector_config = {
        "vector_db_url": os.path.join(
            vector_root, dataset_database.vector_database_name
        ),
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
    }

    graph_db_path = os.path.join(graph_root, dataset_database.graph_database_name)
    graph_config = {
        "graph_database_provider": graph_provider,
        "graph_file_path": graph_db_path,
    }

    # Providers registered through supported_databases expect a URL-style argument.
    if graph_provider != "kuzu":
        graph_config["graph_database_url"] = graph_db_path

    storage_config = {
        "data_root_directory": files_root,
    }

    # Use ContextVar to use these graph and vector configurations are used
    # in the current async context across Cognee
    graph_db_config.set(graph_config)
    vector_db_config.set(vector_config)
    file_storage_config.set(storage_config)
