import asyncio
import re
from time import monotonic
from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.exceptions import DatabaseCredentialsError
from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.infrastructure.databases.dataset_database_handler import (
    DatasetDatabaseHandlerInterface,
)
from cognee.modules.users.models import DatasetDatabase, User


NEO4J_DATASET_DATABASE_HANDLER = "neo4j"
NEO4J_SYSTEM_DATABASE = "system"
NEO4J_DATASET_DATABASE_PREFIX = "cognee"
NEO4J_DATABASE_ONLINE_STATUS = "online"
NEO4J_DATABASE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]{2,62}$")


class Neo4jDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """Handler for per-dataset databases in a local/self-hosted Neo4j DBMS."""

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "neo4j":
            raise ValueError(
                "Neo4jDatasetDatabaseHandler can only be used with Neo4j graph database provider."
            )

        graph_db_name = cls._database_name_for_dataset(dataset_id)

        await cls._create_neo4j_database(graph_db_name)
        await cls._initialize_graph_database(graph_db_name)

        return {
            "graph_database_provider": "neo4j",
            "graph_database_url": graph_config.graph_database_url,
            "graph_database_name": graph_db_name,
            "graph_database_key": graph_config.graph_database_key,
            "graph_dataset_database_handler": NEO4J_DATASET_DATABASE_HANDLER,
            "graph_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        graph_config = get_graph_config()
        connection_info = cls._resolve_neo4j_connection_info(graph_config)

        dataset_database.graph_database_connection_info["graph_database_username"] = (
            connection_info["username"]
        )
        dataset_database.graph_database_connection_info["graph_database_password"] = (
            connection_info["password"]
        )
        dataset_database.graph_database_connection_info["graph_database_allow_anonymous"] = (
            connection_info["allow_anonymous"]
        )

        if not dataset_database.graph_database_url:
            dataset_database.graph_database_url = connection_info["url"]

        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        info = dataset_database.graph_database_connection_info or {}
        graph_db_name = dataset_database.graph_database_name
        graph_database_url = dataset_database.graph_database_url
        graph_database_username = info.get("graph_database_username", "")
        graph_database_password = info.get("graph_database_password", "")
        graph_database_allow_anonymous = info.get("graph_database_allow_anonymous", False)

        evict_graph_engine(
            graph_database_provider="neo4j",
            graph_file_path="",
            graph_database_url=graph_database_url,
            graph_database_name=graph_db_name,
            graph_database_username=graph_database_username,
            graph_database_password=graph_database_password,
            graph_database_allow_anonymous=graph_database_allow_anonymous,
            graph_database_key=dataset_database.graph_database_key,
            graph_dataset_database_handler=NEO4J_DATASET_DATABASE_HANDLER,
        )

        await cls._drop_neo4j_database(graph_db_name)

    @classmethod
    def _database_name_for_dataset(cls, dataset_id: Optional[UUID]) -> str:
        if dataset_id is None:
            raise ValueError("dataset_id is required to create a local Neo4j dataset database.")

        database_name = f"{NEO4J_DATASET_DATABASE_PREFIX}{UUID(str(dataset_id)).hex}"
        cls._validate_database_name(database_name)
        return database_name

    @classmethod
    def _validate_database_name(cls, database_name: str) -> None:
        if not database_name.startswith(NEO4J_DATASET_DATABASE_PREFIX):
            raise ValueError(
                "Refusing to manage a Neo4j database that was not created by the "
                "neo4j dataset handler."
            )

        if not NEO4J_DATABASE_NAME_PATTERN.fullmatch(database_name):
            raise ValueError(
                f"Invalid Neo4j dataset database name: {database_name!r}. "
                "Expected 3-63 lowercase alphanumeric characters starting with a letter."
            )

    @classmethod
    async def _create_neo4j_database(cls, graph_db_name: str) -> None:
        cls._validate_database_name(graph_db_name)
        graph_config = get_graph_config()
        connection_info = cls._resolve_neo4j_connection_info(graph_config)

        driver = cls._create_neo4j_driver(**connection_info)
        try:
            await cls._run_system_query(
                driver,
                f"CREATE DATABASE {graph_db_name} IF NOT EXISTS",
            )
            await cls._wait_for_database_online(driver, graph_db_name)
        finally:
            await cls._close_driver(driver)

    @classmethod
    async def _drop_neo4j_database(cls, graph_db_name: str) -> None:
        cls._validate_database_name(graph_db_name)
        graph_config = get_graph_config()
        connection_info = cls._resolve_neo4j_connection_info(graph_config)

        driver = cls._create_neo4j_driver(**connection_info)
        try:
            await cls._run_system_query(
                driver,
                f"DROP DATABASE {graph_db_name} IF EXISTS",
            )
        finally:
            await cls._close_driver(driver)

    @classmethod
    async def _initialize_graph_database(cls, graph_db_name: str) -> None:
        graph_config = get_graph_config()
        connection_info = cls._resolve_neo4j_connection_info(graph_config)

        engine = create_graph_engine(
            graph_database_provider="neo4j",
            graph_file_path="",
            graph_database_url=connection_info["url"],
            graph_database_name=graph_db_name,
            graph_database_username=connection_info["username"],
            graph_database_password=connection_info["password"],
            graph_database_allow_anonymous=connection_info["allow_anonymous"],
            graph_database_key=graph_config.graph_database_key,
            graph_dataset_database_handler=NEO4J_DATASET_DATABASE_HANDLER,
        )
        await engine.initialize()

    @classmethod
    def _resolve_neo4j_connection_info(cls, graph_config) -> dict:
        graph_database_url = graph_config.graph_database_url
        graph_database_username = graph_config.graph_database_username
        graph_database_password = graph_config.graph_database_password
        graph_database_allow_anonymous = graph_config.graph_database_allow_anonymous

        if not graph_database_url:
            raise EnvironmentError(
                "Missing required GRAPH_DATABASE_URL for local Neo4j multi-user mode."
            )

        if graph_database_username and graph_database_password:
            pass
        elif graph_database_username or graph_database_password:
            provided = "username" if graph_database_username else "password"
            missing = "password" if graph_database_username else "username"
            raise DatabaseCredentialsError(
                message=(
                    f"Neo4j credentials are incomplete: '{provided}' was provided but "
                    f"'{missing}' is missing. Please provide both GRAPH_DATABASE_USERNAME "
                    "and GRAPH_DATABASE_PASSWORD, or neither."
                ),
            )
        elif not graph_database_allow_anonymous:
            raise DatabaseCredentialsError(
                message=(
                    "Neo4j credentials not provided. Set GRAPH_DATABASE_USERNAME and "
                    "GRAPH_DATABASE_PASSWORD, or set GRAPH_DATABASE_ALLOW_ANONYMOUS=true."
                ),
            )

        return {
            "url": graph_database_url,
            "username": graph_database_username,
            "password": graph_database_password,
            "allow_anonymous": graph_database_allow_anonymous,
        }

    @classmethod
    def _create_neo4j_driver(
        cls,
        url: str,
        username: str,
        password: str,
        allow_anonymous: bool,
    ):
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as error:
            raise ImportError(
                "Neo4j local dataset database handling requires the neo4j extra. "
                "Install cognee with `pip install cognee[neo4j]` or `uv sync --extra neo4j`."
            ) from error

        auth = (username, password) if username and password else None
        if auth is None and not allow_anonymous:
            raise DatabaseCredentialsError(
                message=(
                    "Neo4j credentials not provided. Set GRAPH_DATABASE_USERNAME and "
                    "GRAPH_DATABASE_PASSWORD, or set GRAPH_DATABASE_ALLOW_ANONYMOUS=true."
                ),
            )

        return AsyncGraphDatabase.driver(
            url,
            auth=auth,
            max_connection_lifetime=120,
            notifications_min_severity="OFF",
            keep_alive=True,
        )

    @classmethod
    async def _run_system_query(cls, driver, query: str, params: Optional[dict] = None) -> list:
        try:
            async with driver.session(database=NEO4J_SYSTEM_DATABASE) as session:
                result = await session.run(query, parameters=params or {})
                return await result.data()
        except Exception as error:
            if cls._is_neo4j_error(error):
                raise EnvironmentError(
                    "Local Neo4j multi-user mode requires a Neo4j deployment that supports "
                    "CREATE/DROP DATABASE and credentials with database-management privileges."
                ) from error
            raise

    @classmethod
    async def _wait_for_database_online(
        cls,
        driver,
        graph_db_name: str,
        timeout_seconds: int = 30,
    ) -> None:
        deadline = monotonic() + timeout_seconds
        last_status = "unknown"

        while monotonic() < deadline:
            records = await cls._run_system_query(
                driver,
                (
                    "SHOW DATABASES YIELD name, currentStatus "
                    "WHERE name = $database_name "
                    "RETURN currentStatus"
                ),
                {"database_name": graph_db_name},
            )

            if records:
                last_status = records[0].get("currentStatus", last_status)
                if last_status == NEO4J_DATABASE_ONLINE_STATUS:
                    return

            await asyncio.sleep(1)

        raise TimeoutError(
            f"Neo4j dataset database '{graph_db_name}' did not become online within "
            f"{timeout_seconds} seconds. Last status: {last_status}."
        )

    @classmethod
    async def _close_driver(cls, driver) -> None:
        close_result = driver.close()
        if asyncio.iscoroutine(close_result):
            await close_result

    @classmethod
    def _is_neo4j_error(cls, error: Exception) -> bool:
        try:
            from neo4j.exceptions import Neo4jError
        except ImportError:
            return False

        return isinstance(error, Neo4jError)
