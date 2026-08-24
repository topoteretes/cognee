from types import SimpleNamespace
from uuid import UUID

import pytest

from cognee.context_global_variables import multi_user_support_possible
from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
    supported_dataset_database_handlers,
)
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jCommunityDatasetDatabaseHandler import (
    NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER,
    Neo4jCommunityDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.neo4j_driver import neo4j_community_containers
from cognee.infrastructure.databases.graph.neo4j_driver.neo4j_community_containers import (
    Neo4jCommunityContainerManager,
    bolt_url_for_port,
    container_name_for_dataset,
    volume_name_for_dataset,
)
from cognee.infrastructure.databases.vector.config import get_vectordb_config


DATASET_ID = UUID("12345678-1234-5678-1234-567812345678")
CONTAINER_NAME = "cognee-neo4j-12345678123456781234567812345678"
VOLUME_NAME = "cognee-neo4j-data-12345678123456781234567812345678"


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
        "NEO4J_COMMUNITY_MAX_CONTAINERS",
    ):
        monkeypatch.delenv(env_name, raising=False)

    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()
    yield
    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()


def configure_neo4j_community(monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER)
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "lancedb")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")
    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()


class FakeContainerManager:
    def __init__(self):
        self.created = []
        self.ensured = []
        self.removed = []
        self.fail_ports = set()

    async def create_container(self, dataset_id, container_name, volume_name, host_port, password):
        if host_port in self.fail_ports:
            raise RuntimeError("Bind for 127.0.0.1 failed: port is already allocated")
        self.created.append(
            {
                "dataset_id": dataset_id,
                "container_name": container_name,
                "volume_name": volume_name,
                "host_port": host_port,
                "password": password,
            }
        )

    async def ensure_running(self, container_name, host_port):
        self.ensured.append((container_name, host_port))

    async def remove_container(self, container_name, volume_name, bolt_url=None):
        self.removed.append((container_name, volume_name, bolt_url))


@pytest.fixture
def fake_manager(monkeypatch):
    manager = FakeContainerManager()
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "Neo4jCommunityDatasetDatabaseHandler.get_container_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "Neo4jCommunityDatasetDatabaseHandler.check_docker_available",
        lambda: (True, "Docker daemon is running."),
    )
    return manager


def test_neo4j_community_handler_is_registered():
    handler = supported_dataset_database_handlers[NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER]

    assert handler["handler_instance"] is Neo4jCommunityDatasetDatabaseHandler
    assert handler["handler_provider"] == "neo4j"


def test_neo4j_community_handler_enables_multi_user_support(monkeypatch):
    configure_neo4j_community(monkeypatch)

    assert (
        get_graph_config().graph_dataset_database_handler
        == NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER
    )
    assert multi_user_support_possible() is True


def test_container_and_volume_names_are_deterministic():
    assert container_name_for_dataset(DATASET_ID) == CONTAINER_NAME
    assert container_name_for_dataset(str(DATASET_ID)) == CONTAINER_NAME
    assert volume_name_for_dataset(DATASET_ID) == VOLUME_NAME


@pytest.mark.asyncio
async def test_create_dataset_provisions_container_and_encrypts_password(monkeypatch, fake_manager):
    configure_neo4j_community(monkeypatch)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "Neo4jCommunityDatasetDatabaseHandler.allocate_free_port",
        lambda: 7799,
    )

    database_info = await Neo4jCommunityDatasetDatabaseHandler.create_dataset(DATASET_ID, None)

    assert len(fake_manager.created) == 1
    created = fake_manager.created[0]
    assert created["container_name"] == CONTAINER_NAME
    assert created["volume_name"] == VOLUME_NAME
    assert created["host_port"] == 7799

    assert database_info["graph_database_provider"] == "neo4j"
    assert database_info["graph_database_url"] == "bolt://localhost:7799"
    assert database_info["graph_database_name"] == "neo4j"
    assert (
        database_info["graph_dataset_database_handler"] == NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER
    )

    connection_info = database_info["graph_database_connection_info"]
    assert connection_info["graph_database_username"] == "neo4j"
    assert connection_info["container_name"] == CONTAINER_NAME
    assert connection_info["volume_name"] == VOLUME_NAME
    assert connection_info["host_port"] == 7799

    # The plaintext password reached the container but is stored encrypted.
    plaintext = created["password"]
    stored = connection_info["graph_database_password"]
    assert stored != plaintext
    cipher = Neo4jCommunityDatasetDatabaseHandler._get_cipher()
    assert cipher.decrypt(stored.encode()).decode() == plaintext


@pytest.mark.asyncio
async def test_create_dataset_retries_on_port_collision(monkeypatch, fake_manager):
    configure_neo4j_community(monkeypatch)
    ports = iter([7001, 7002, 7003])
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "Neo4jCommunityDatasetDatabaseHandler.allocate_free_port",
        lambda: next(ports),
    )
    fake_manager.fail_ports = {7001, 7002}

    database_info = await Neo4jCommunityDatasetDatabaseHandler.create_dataset(DATASET_ID, None)

    assert database_info["graph_database_url"] == "bolt://localhost:7003"
    assert [c["host_port"] for c in fake_manager.created] == [7003]
    # Failed attempts clean up after themselves.
    assert len(fake_manager.removed) == 2


@pytest.mark.asyncio
async def test_create_dataset_requires_neo4j_provider(monkeypatch, fake_manager):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "kuzu")
    get_graph_config.cache_clear()

    with pytest.raises(ValueError, match="Neo4j graph database provider"):
        await Neo4jCommunityDatasetDatabaseHandler.create_dataset(DATASET_ID, None)


@pytest.mark.asyncio
async def test_create_dataset_fails_actionably_without_docker(monkeypatch):
    configure_neo4j_community(monkeypatch)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "Neo4jCommunityDatasetDatabaseHandler.check_docker_available",
        lambda: (False, "Docker CLI not found on PATH."),
    )

    with pytest.raises(RuntimeError, match="Docker CLI not found"):
        await Neo4jCommunityDatasetDatabaseHandler.create_dataset(DATASET_ID, None)


@pytest.mark.asyncio
async def test_resolve_decrypts_password_and_starts_container(monkeypatch, fake_manager):
    configure_neo4j_community(monkeypatch)
    cipher = Neo4jCommunityDatasetDatabaseHandler._get_cipher()
    dataset_database = SimpleNamespace(
        graph_database_url="bolt://localhost:7799",
        graph_database_connection_info={
            "graph_database_username": "neo4j",
            "graph_database_password": cipher.encrypt(b"secret-pw").decode(),
            "container_name": CONTAINER_NAME,
            "volume_name": VOLUME_NAME,
            "host_port": 7799,
        },
    )

    resolved = await Neo4jCommunityDatasetDatabaseHandler.resolve_dataset_connection_info(
        dataset_database
    )

    assert resolved.graph_database_connection_info["graph_database_password"] == "secret-pw"
    assert fake_manager.ensured == [(CONTAINER_NAME, 7799)]


@pytest.mark.asyncio
async def test_delete_dataset_evicts_engines_and_removes_container(monkeypatch, fake_manager):
    configure_neo4j_community(monkeypatch)
    evicted_urls = []

    async def fake_aevict(url):
        evicted_urls.append(url)
        return 1

    # The `graph` package re-exports a `get_graph_engine` function that shadows
    # the module of the same name, so resolve the module via importlib.
    # EngineCacheOps uses __slots__, so patch the module-level instance with a
    # stand-in rather than setting an attribute on it.
    import importlib

    get_graph_engine_module = importlib.import_module(
        "cognee.infrastructure.databases.graph.get_graph_engine"
    )
    monkeypatch.setattr(
        get_graph_engine_module,
        "graph_engine_cache",
        SimpleNamespace(aevict_for_url=fake_aevict),
    )

    dataset_database = SimpleNamespace(
        dataset_id=DATASET_ID,
        graph_database_url="bolt://localhost:7799",
        graph_database_connection_info={
            "container_name": CONTAINER_NAME,
            "volume_name": VOLUME_NAME,
            "host_port": 7799,
        },
    )

    await Neo4jCommunityDatasetDatabaseHandler.delete_dataset(dataset_database)

    assert evicted_urls == ["bolt://localhost:7799"]
    assert fake_manager.removed == [(CONTAINER_NAME, VOLUME_NAME, "bolt://localhost:7799")]


class DockerCall:
    """Recorded docker CLI invocation with a scripted response."""

    def __init__(self, args, returncode=0, stdout="", stderr=""):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeDocker:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def run(self, args, timeout):
        self.calls.append(args)
        key = tuple(args[:2]) if args[0] in ("volume",) else (args[0],)
        response = self.responses.get(key, SimpleNamespace(returncode=0, out="", err=""))
        return SimpleNamespace(
            returncode=response.returncode,
            stdout=response.out.encode(),
            stderr=response.err.encode(),
        )


def make_fake_docker(monkeypatch, responses=None):
    fake = FakeDocker(responses)
    monkeypatch.setattr(neo4j_community_containers, "_run_docker_sync", fake.run)
    return fake


@pytest.mark.asyncio
async def test_manager_ensure_running_starts_stopped_container(monkeypatch):
    manager = Neo4jCommunityContainerManager()
    fake = make_fake_docker(
        monkeypatch,
        responses={
            ("inspect",): SimpleNamespace(returncode=0, out="exited\n", err=""),
            ("ps",): SimpleNamespace(returncode=0, out="", err=""),
        },
    )
    monkeypatch.setattr(neo4j_community_containers, "_bolt_ready", lambda port, timeout=2.0: True)

    await manager.ensure_running(CONTAINER_NAME, 7799)

    commands = [call[0] for call in fake.calls]
    assert "inspect" in commands
    assert "start" in commands
    assert manager.container_for_url(bolt_url_for_port(7799)) == CONTAINER_NAME


@pytest.mark.asyncio
async def test_manager_ensure_running_skips_start_when_bolt_answers(monkeypatch):
    manager = Neo4jCommunityContainerManager()
    fake = make_fake_docker(monkeypatch)
    monkeypatch.setattr(neo4j_community_containers, "_bolt_ready", lambda port, timeout=2.0: True)
    manager._touch(CONTAINER_NAME, bolt_url_for_port(7799))

    await manager.ensure_running(CONTAINER_NAME, 7799)

    assert fake.calls == []


@pytest.mark.asyncio
async def test_manager_ensure_running_errors_when_container_is_missing(monkeypatch):
    manager = Neo4jCommunityContainerManager()
    make_fake_docker(
        monkeypatch,
        responses={
            ("inspect",): SimpleNamespace(returncode=1, out="", err="No such object"),
        },
    )
    monkeypatch.setattr(neo4j_community_containers, "_bolt_ready", lambda port, timeout=2.0: False)

    with pytest.raises(RuntimeError, match="does not exist"):
        await manager.ensure_running(CONTAINER_NAME, 7799)


@pytest.mark.asyncio
async def test_manager_enforces_container_ceiling(monkeypatch):
    monkeypatch.setenv("NEO4J_COMMUNITY_MAX_CONTAINERS", "2")
    manager = Neo4jCommunityContainerManager()
    running = "cognee-neo4j-aaa\ncognee-neo4j-bbb\n"
    fake = make_fake_docker(
        monkeypatch,
        responses={
            ("ps",): SimpleNamespace(returncode=0, out=running, err=""),
        },
    )
    monkeypatch.setattr(neo4j_community_containers, "_bolt_ready", lambda port, timeout=2.0: True)
    # aaa was used more recently than bbb, so bbb is the LRU victim... but an
    # unknown container is treated as oldest; make both known with bbb older.
    manager._touch("cognee-neo4j-bbb", bolt_url_for_port(7001))
    manager._touch("cognee-neo4j-aaa", bolt_url_for_port(7002))

    await manager.create_container(
        dataset_id=DATASET_ID,
        container_name=CONTAINER_NAME,
        volume_name=VOLUME_NAME,
        host_port=7799,
        password="pw",
    )

    stop_calls = [call for call in fake.calls if call[0] == "stop"]
    assert stop_calls == [["stop", "cognee-neo4j-bbb"]]
    run_calls = [call for call in fake.calls if call[0] == "run"]
    assert len(run_calls) == 1


@pytest.mark.asyncio
async def test_manager_stop_container_for_url_is_noop_for_unknown_url(monkeypatch):
    manager = Neo4jCommunityContainerManager()
    fake = make_fake_docker(monkeypatch)

    await manager.stop_container_for_url("bolt://localhost:9999")

    assert fake.calls == []


@pytest.mark.asyncio
async def test_manager_stop_container_for_url_stops_known_container(monkeypatch):
    manager = Neo4jCommunityContainerManager()
    fake = make_fake_docker(monkeypatch)
    manager._touch(CONTAINER_NAME, bolt_url_for_port(7799))

    await manager.stop_container_for_url(bolt_url_for_port(7799))

    assert fake.calls == [["stop", CONTAINER_NAME]]
    # Second stop for the same url is a no-op (registry entry is gone).
    await manager.stop_container_for_url(bolt_url_for_port(7799))
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_manager_remove_container_removes_container_and_volume(monkeypatch):
    manager = Neo4jCommunityContainerManager()
    fake = make_fake_docker(monkeypatch)
    manager._touch(CONTAINER_NAME, bolt_url_for_port(7799))

    await manager.remove_container(CONTAINER_NAME, VOLUME_NAME, bolt_url_for_port(7799))

    assert ["rm", "--force", CONTAINER_NAME] in fake.calls
    assert ["volume", "rm", "--force", VOLUME_NAME] in fake.calls
    assert manager.container_for_url(bolt_url_for_port(7799)) is None


@pytest.mark.asyncio
async def test_community_adapter_close_stops_container(monkeypatch):
    pytest.importorskip("neo4j")
    from cognee.infrastructure.databases.graph.neo4j_driver.neo4j_community_adapter import (
        Neo4jCommunityAdapter,
    )

    manager = Neo4jCommunityContainerManager()
    fake = make_fake_docker(monkeypatch)
    manager._touch(CONTAINER_NAME, bolt_url_for_port(7799))
    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver."
        "neo4j_community_adapter.get_container_manager",
        lambda: manager,
    )

    class FakeDriver:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    driver = FakeDriver()
    adapter = Neo4jCommunityAdapter(
        graph_database_url=bolt_url_for_port(7799),
        graph_database_username="neo4j",
        graph_database_password="pw",
        graph_database_name="neo4j",
        driver=driver,
    )

    await adapter.close()

    assert driver.closed is True
    assert fake.calls == [["stop", CONTAINER_NAME]]


def test_create_graph_engine_selects_community_adapter(monkeypatch):
    pytest.importorskip("neo4j")
    from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
    from cognee.infrastructure.databases.graph.neo4j_driver.neo4j_community_adapter import (
        Neo4jCommunityAdapter,
    )

    engine = create_graph_engine(
        graph_database_provider="neo4j",
        graph_file_path="",
        graph_database_url="bolt://localhost:7799",
        graph_database_name="neo4j",
        graph_database_username="neo4j",
        graph_database_password="pw",
        graph_dataset_database_handler=NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER,
    )
    try:
        assert isinstance(engine, Neo4jCommunityAdapter)
    finally:
        from cognee.infrastructure.databases.graph.get_graph_engine import graph_engine_cache

        graph_engine_cache.evict(
            graph_database_provider="neo4j",
            graph_file_path="",
            graph_database_url="bolt://localhost:7799",
            graph_database_name="neo4j",
            graph_database_username="neo4j",
            graph_database_password="pw",
            graph_dataset_database_handler=NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER,
        )


def test_max_running_containers_env_override(monkeypatch):
    monkeypatch.setenv("NEO4J_COMMUNITY_MAX_CONTAINERS", "3")
    assert neo4j_community_containers.get_max_running_containers() == 3

    monkeypatch.delenv("NEO4J_COMMUNITY_MAX_CONTAINERS")
    assert neo4j_community_containers.get_max_running_containers() >= 1
