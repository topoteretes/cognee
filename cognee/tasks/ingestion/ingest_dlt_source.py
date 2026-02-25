from typing import List

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational.config import get_relational_config
import dlt


async def ingest_dlt_source(
    dlt_source,
    dataset_name: str,
) -> List[str]:
    """
    Ingests a dlt (re)source by running the dlt pipeline on it.
    The pipeline destination is an SQL database, SQLite or Postgres depending on the cognee configuration.

    Parameters:
        dlt_source (str): The name of the dlt (re)source.
        dataset_name (str): The name of the dataset.
    Returns:
        List[str]: The schema of the ingested database, as a list.
    """

    relational_config = get_relational_config()
    dlt_db_name = f"dlt_database_{dataset_name}"

    # TODO: For now works only for postgres, I will add sqlite tomorrow, had to catch the flight and couldn't debug til the end
    # if relational_config.db_provider == "postgres":

    # Create postgres database for dlt data
    await _create_pg_database(dlt_db_name)

    # Set up postgres as the dlt destination
    # dlt's postgres destination uses psycopg2, which expects a sync
    # postgresql:// URL (not postgresql+asyncpg://).
    dlt_connection_str = f"postgresql://{relational_config.db_username}:{relational_config.db_password}@{relational_config.db_host}:{int(relational_config.db_port)}/{dlt_db_name}"
    destination = dlt.destinations.postgres(dlt_connection_str)

    # Execute dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="ingest_dlt_source",
        destination=destination,
        dataset_name=dataset_name,
    )

    # TODO: In an example, where I create data for ingestion, I know the primary key for merging, what if I don't know it?
    pipeline.run(dlt_source, write_disposition="merge", primary_key="id")

    from cognee.infrastructure.databases.relational.get_migration_relational_engine import (
        get_migration_relational_engine,
    )
    from cognee.infrastructure.databases.relational.config import get_migration_config

    migration_config = get_migration_config()
    migration_config.migration_db_provider = relational_config.db_provider
    migration_config.migration_db_name = dlt_db_name
    migration_config.migration_db_host = relational_config.db_host
    migration_config.migration_db_port = relational_config.db_port
    migration_config.migration_db_username = relational_config.db_username
    migration_config.migration_db_password = relational_config.db_password

    engine = get_migration_relational_engine()
    schema = await engine.extract_schema()
    # Remove keys from schema which have "dlt" and "staging" in them TODO: maybe find a more robust way of doing this
    schema_keys = list(schema.keys())
    for key in schema_keys:
        if "dlt" in key or "staging" in key:
            schema.pop(key, None)
    import json

    schema_str = json.dumps(schema)

    return [schema_str]


async def _create_pg_database(db_name):
    relational_config = get_relational_config()
    maintenance_db_name = "postgres"
    maintenance_db_url = URL.create(
        "postgresql+asyncpg",
        username=relational_config.db_username,
        password=relational_config.db_password,
        host=relational_config.db_host,
        port=int(relational_config.db_port),
        database=maintenance_db_name,
    )
    maintenance_engine = create_async_engine(maintenance_db_url)

    # Connect to maintenance db in order to create new database
    # Make sure to execute CREATE DATABASE outside of transaction block, and set AUTOCOMMIT isolation level
    connection = await maintenance_engine.connect()
    connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
    exists_result = await connection.execute(
        text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
        {"db_name": db_name},
    )
    if exists_result.scalar() is None:
        await connection.execute(text(f'CREATE DATABASE "{db_name}";'))

    # Clean up resources
    await connection.close()


async def migrate_dlt_database(data: List[Data]):
    from cognee.tasks.ingestion.migrate_relational_database import migrate_relational_database
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
    from cognee.infrastructure.files.utils.open_data_file import open_data_file
    import json

    graph_engine = await get_graph_engine()

    data_object = data[0]

    async def read_json_file(path: str):
        async with open_data_file(path, "r", encoding="utf-8") as f:
            return json.load(f)

    schema = await read_json_file(data_object.raw_data_location)

    await migrate_relational_database(graph_db=graph_engine, schema=schema)
