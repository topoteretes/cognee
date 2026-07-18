from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.infrastructure.databases.relational import get_relational_config
from cognee.infrastructure.databases.postgres import (
    create_pg_schema_if_not_exists,
    drop_pg_schema_if_exists,
    dataset_schema_name,
)
from cognee.modules.users.models import User, DatasetDatabase


class PostgresGraphSharedDatasetDatabaseHandler:
    """Per-dataset Postgres graph isolation via a schema in one shared database.

    The default ``PostgresGraphDatasetDatabaseHandler`` provisions a whole
    Postgres database per dataset. This handler instead keeps each dataset's
    ``graph_node``/``graph_edge`` tables in a dedicated schema
    (``ds_<dataset_id>``) of cognee's main Postgres database, isolating datasets
    by pinning the per-dataset graph engine's ``search_path`` to that schema
    (see ``PostgresAdapter``). The shared database is cognee's relational
    database, so credentials/host/port come from the relational configuration.

    Selected with ``GRAPH_DATASET_DATABASE_HANDLER=postgres_graph_shared``
    (requires ``GRAPH_DATABASE_PROVIDER=postgres`` and
    ``ENABLE_BACKEND_ACCESS_CONTROL=true``).
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "postgres":
            raise ValueError(
                "PostgresGraphSharedDatasetDatabaseHandler can only be used "
                "with the postgres graph database provider."
            )

        relational_config = get_relational_config()
        schema = dataset_schema_name(dataset_id)
        db_name = relational_config.db_name
        host = relational_config.db_host
        port = relational_config.db_port

        await create_pg_schema_if_not_exists(
            db_name,
            schema,
            host=host,
            port=port,
            username=relational_config.db_username,
            password=relational_config.db_password,
        )

        # Create the graph tables inside the dataset schema. The engine is pinned
        # to ``schema`` via search_path, so initialize() emits graph_node/graph_edge
        # into that schema (and nowhere else).
        engine = create_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_name=db_name,
            graph_database_username=relational_config.db_username,
            graph_database_password=relational_config.db_password,
            graph_database_host=host,
            graph_database_port=port,
            graph_dataset_database_handler=graph_config.graph_dataset_database_handler,
            graph_database_schema=schema,
        )
        await engine.initialize()

        return {
            "graph_database_provider": "postgres",
            "graph_database_url": "",
            "graph_database_name": db_name,
            "graph_database_key": graph_config.graph_database_key,
            "graph_dataset_database_handler": "postgres_graph_shared",
            "graph_database_connection_info": {
                "graph_database_host": host,
                "graph_database_port": port,
                "graph_database_schema": schema,
            },
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # Credentials are never persisted; pull them from the live relational
        # config (the shared database backing this mode).
        relational_config = get_relational_config()
        dataset_database.graph_database_connection_info["graph_database_username"] = (
            relational_config.db_username
        )
        dataset_database.graph_database_connection_info["graph_database_password"] = (
            relational_config.db_password
        )
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        info = dataset_database.graph_database_connection_info
        host = info.get("graph_database_host", "")
        port = info.get("graph_database_port", "")
        username = info.get("graph_database_username", "")
        password = info.get("graph_database_password", "")
        schema = info["graph_database_schema"]
        db_name = dataset_database.graph_database_name

        evict_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_name=db_name,
            graph_database_username=username,
            graph_database_password=password,
            graph_database_host=host,
            graph_database_port=port,
            graph_dataset_database_handler=dataset_database.graph_dataset_database_handler,
            graph_database_schema=schema,
        )

        await drop_pg_schema_if_exists(
            db_name,
            schema,
            host=host,
            port=port,
            username=username,
            password=password,
        )
