from types import SimpleNamespace
from uuid import UUID

import pytest

from cognee.context_global_variables import (
    multi_user_support_possible,
)
from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
    supported_dataset_database_handlers,
)
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jDatasetDatabaseHandler import (
    NEO4J_DATASET_DATABASE_HANDLER,
    Neo4jDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.vector.config import get_vectordb_config


DATASET_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture(autouse=True)
def clear_database_config(monkeypatch):
    for env_name in (
        "GRAPH_DATABASE_PROVIDER",
        "GRAPH_DATASET_DATABASE_HANDLER",
        "GRAPH_DATABASE_URL",
        "GRAPH_DATABASE_USERNAME",
        "GRAPH_DATABASE_PASSWORD",
        "GRAPH_DATABASE_ALLOW_ANONYMOUS",
        "VECTOR_DB_PROVIDER",
        "VECTOR_DATASET_DATABASE_HANDLER",
    ):
        monkeypatch.delenv(env_name, raising=False)

    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()
    yield
    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()


def configure_neo4j(monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", NEO4J_DATASET_DATABASE_HANDLER)
    monkeypatch.setenv("GRAPH_DATABASE_URL", "bolt://localhost:7687")
    monkeypatch.setenv("GRAPH_DATABASE_USERNAME", "neo4j")
    monkeypatch.setenv("GRAPH_DATABASE_PASSWORD", "pleaseletmein")
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "lancedb")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")
    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()


def test_neo4j_handler_is_registered():
    handler = supported_dataset_database_handlers[NEO4J_DATASET_DATABASE_HANDLER]

    assert handler["handler_instance"] is Neo4jDatasetDatabaseHandler
    assert handler["handler_provider"] == "neo4j"


def test_neo4j_dataset_handler_enables_multi_user_support(monkeypatch):
    configure_neo4j(monkeypatch)

    assert get_graph_config().graph_dataset_database_handler == NEO4J_DATASET_DATABASE_HANDLER
    assert multi_user_support_possible() is True


def test_neo4j_dataset_handler_alias_maps_to_dataset_handler(monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", "neo4j")
    get_graph_config.cache_clear()

    assert get_graph_config().graph_dataset_database_handler == NEO4J_DATASET_DATABASE_HANDLER


@pytest.mark.asyncio
async def test_create_dataset_creates_neo4j_database(monkeypatch):
    configure_neo4j(monkeypatch)
    created_databases = []
    initialized_databases = []

    async def fake_create_neo4j_database(cls, graph_db_name):
        created_databases.append(graph_db_name)

    async def fake_initialize_graph_database(cls, graph_db_name):
        initialized_databases.append(graph_db_name)

    monkeypatch.setattr(
        Neo4jDatasetDatabaseHandler,
        "_create_neo4j_database",
        classmethod(fake_create_neo4j_database),
    )
    monkeypatch.setattr(
        Neo4jDatasetDatabaseHandler,
        "_initialize_graph_database",
        classmethod(fake_initialize_graph_database),
    )

    database_info = await Neo4jDatasetDatabaseHandler.create_dataset(DATASET_ID, None)

    assert created_databases == ["cognee12345678123456781234567812345678"]
    assert initialized_databases == created_databases
    assert database_info == {
        "graph_database_provider": "neo4j",
        "graph_database_url": "bolt://localhost:7687",
        "graph_database_name": "cognee12345678123456781234567812345678",
        "graph_database_key": "",
        "graph_dataset_database_handler": NEO4J_DATASET_DATABASE_HANDLER,
        "graph_database_connection_info": {},
    }


@pytest.mark.asyncio
async def test_resolve_dataset_connection_info_uses_live_config_credentials(monkeypatch):
    configure_neo4j(monkeypatch)
    dataset_database = SimpleNamespace(
        graph_database_url="bolt://stored:7687",
        graph_database_connection_info={},
    )

    resolved = await Neo4jDatasetDatabaseHandler.resolve_dataset_connection_info(dataset_database)

    assert resolved.graph_database_url == "bolt://stored:7687"
    assert resolved.graph_database_connection_info == {
        "graph_database_username": "neo4j",
        "graph_database_password": "pleaseletmein",
        "graph_database_allow_anonymous": False,
    }


@pytest.mark.asyncio
async def test_create_neo4j_database_uses_system_database(monkeypatch):
    configure_neo4j(monkeypatch)
    calls = []
    closed = []

    class FakeResult:
        async def data(self):
            return []

    class FakeSession:
        def __init__(self, database):
            self.database = database

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def run(self, query, parameters=None):
            calls.append((self.database, query, parameters))
            return FakeResult()

    class FakeDriver:
        def session(self, database):
            return FakeSession(database)

        async def close(self):
            closed.append(True)

    async def fake_wait_for_database_online(cls, driver, graph_db_name, timeout_seconds=30):
        calls.append(("wait", graph_db_name, timeout_seconds))

    monkeypatch.setattr(
        Neo4jDatasetDatabaseHandler,
        "_create_neo4j_driver",
        classmethod(lambda cls, **connection_info: FakeDriver()),
    )
    monkeypatch.setattr(
        Neo4jDatasetDatabaseHandler,
        "_wait_for_database_online",
        classmethod(fake_wait_for_database_online),
    )

    await Neo4jDatasetDatabaseHandler._create_neo4j_database(
        "cognee12345678123456781234567812345678"
    )

    assert calls == [
        (
            "system",
            "CREATE DATABASE cognee12345678123456781234567812345678 IF NOT EXISTS",
            {},
        ),
        ("wait", "cognee12345678123456781234567812345678", 30),
    ]
    assert closed == [True]


@pytest.mark.asyncio
async def test_delete_dataset_evicts_and_drops_database(monkeypatch):
    configure_neo4j(monkeypatch)
    dropped_databases = []
    evicted_configs = []

    async def fake_drop_neo4j_database(cls, graph_db_name):
        dropped_databases.append(graph_db_name)

    monkeypatch.setattr(
        Neo4jDatasetDatabaseHandler,
        "_drop_neo4j_database",
        classmethod(fake_drop_neo4j_database),
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "Neo4jDatasetDatabaseHandler.evict_graph_engine",
        lambda **kwargs: evicted_configs.append(kwargs),
    )

    dataset_database = SimpleNamespace(
        graph_database_name="cognee12345678123456781234567812345678",
        graph_database_url="bolt://localhost:7687",
        graph_database_key="",
        graph_database_connection_info={},
    )

    await Neo4jDatasetDatabaseHandler.delete_dataset(dataset_database)

    assert dropped_databases == ["cognee12345678123456781234567812345678"]
    assert evicted_configs == [
        {
            "graph_database_provider": "neo4j",
            "graph_file_path": "",
            "graph_database_url": "bolt://localhost:7687",
            "graph_database_name": "cognee12345678123456781234567812345678",
            "graph_database_username": "neo4j",
            "graph_database_password": "pleaseletmein",
            "graph_database_allow_anonymous": False,
            "graph_database_key": "",
            "graph_dataset_database_handler": NEO4J_DATASET_DATABASE_HANDLER,
        }
    ]
