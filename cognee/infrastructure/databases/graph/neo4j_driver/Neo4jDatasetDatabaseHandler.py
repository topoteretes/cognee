import asyncio
import re
from time import monotonic
from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.exceptions import (
    DatabaseCredentialsError,
    Neo4jMultiDatabaseSupportError,
)
from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    graph_engine_cache,
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
NEO4J_ENTERPRISE_EDITION = "enterprise"
NEO4J_EDITION_QUERY = "CALL dbms.components() YIELD edition RETURN edition"

# Multi-database management is Enterprise/Aura only, so this handler cannot run
# against Community edition; these are the ways out we can point users to.
NEO4J_MULTI_DATABASE_OPTIONS = (
    "Per-dataset graph isolation on Neo4j requires multi-database support "
    "(CREATE DATABASE), which is available on Neo4j Enterprise and AuraDB only. "
    "Options: "
    "(1) connect to a Neo4j Enterprise or AuraDB deployment; "
    "(2) keep this server and set GRAPH_DATASET_DATABASE_HANDLER=neo4j_community "
    "to isolate each dataset in its own Docker container (requires a reachable "
    "Docker daemon); "
    "(3) switch to a graph database with built-in multi-tenant support, e.g. the "
    "default ladybug/kuzu backend; "
    "(4) set ENABLE_BACKEND_ACCESS_CONTROL=false to run without per-dataset "
    "databases — all datasets then share a single graph database and per-dataset "
    "isolation is lost."
)


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

        graph_engine_cache.evict(
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
            await cls._ensure_multi_database_support(driver)
            await cls._run_system_query(
                driver,
                f"CREATE DATABASE {graph_db_name} IF NOT EXISTS",
            )
            await cls._wait_for_database_online(driver, graph_db_name)
        finally:
            await cls._close_driver(driver)

    @classmethod
    async def _ensure_multi_database_support(cls, driver) -> None:
        """Fail before CREATE DATABASE on editions that cannot host per-dataset databases.

        The edition probe is best-effort: if the server restricts
        ``dbms.components()`` the probe is skipped, and the error translation in
        ``_run_system_query`` catches the CREATE DATABASE failure instead.
        """
        try:
            records = await cls._run_system_query(driver, NEO4J_EDITION_QUERY)
        except Exception:
            return

        edition = records[0].get("edition", "") if records else ""
        if edition and edition.lower() != NEO4J_ENTERPRISE_EDITION:
            raise cls._multi_database_support_error(
                f"The configured Neo4j server reports the '{edition}' edition, "
                "which supports only a single database."
            )

    @classmethod
    def _multi_database_support_error(cls, detail: str) -> Neo4jMultiDatabaseSupportError:
        return Neo4jMultiDatabaseSupportError(message=f"{detail} {NEO4J_MULTI_DATABASE_OPTIONS}")

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
                raise cls._translate_neo4j_admin_error(error) from error
            raise

    @classmethod
    def _translate_neo4j_admin_error(cls, error: Exception) -> Exception:
        """Map a raw Neo4jError from a system-database command to an actionable error.

        Community edition rejects CREATE/DROP DATABASE with
        ``Neo.ClientError.Statement.UnsupportedAdministrationCommand``; security
        codes mean the credentials lack database-management privileges.
        """
        code = getattr(error, "code", None) or ""

        if "UnsupportedAdministrationCommand" in code:
            return cls._multi_database_support_error(
                "The configured Neo4j server rejected the database-management "
                f"command as unsupported ({code}), which typically means it runs "
                "the Community edition."
            )

        if ".Security." in code:
            return DatabaseCredentialsError(
                message=(
                    f"Neo4j rejected the database-management command ({code}). "
                    "Local Neo4j multi-user mode requires credentials with "
                    "database-management privileges (e.g. the admin role). Update "
                    "GRAPH_DATABASE_USERNAME/GRAPH_DATABASE_PASSWORD to a user "
                    "that can run CREATE/DROP DATABASE."
                ),
            )

        return EnvironmentError(
            "Local Neo4j multi-user mode requires a Neo4j deployment that supports "
            "CREATE/DROP DATABASE and credentials with database-management privileges."
        )

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
