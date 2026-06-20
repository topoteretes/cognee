import asyncio
import hashlib
from types import SimpleNamespace
from uuid import UUID

import pytest

from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
    supported_dataset_database_handlers,
)
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jAuraDevDatasetDatabaseHandler import (
    NEO4J_AURA_INSTANCE_NAME_LIMIT,
    NEO4J_AURA_INSTANCE_NAME_PREFIX,
    Neo4jAuraDevDatasetDatabaseHandler,
    _base62_encode,
)


DATASET_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture(autouse=True)
def clear_graph_config(monkeypatch):
    """Reset cached graph config around each test so env changes take effect."""
    for env_name in (
        "GRAPH_DATABASE_PROVIDER",
        "GRAPH_DATASET_DATABASE_HANDLER",
        "NEO4J_CLIENT_ID",
        "NEO4J_CLIENT_SECRET",
        "NEO4J_TENANT_ID",
        "NEO4J_ENCRYPTION_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", "neo4j_aura_dev")
    monkeypatch.setenv("NEO4J_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("NEO4J_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("NEO4J_TENANT_ID", "test-tenant-id")

    get_graph_config.cache_clear()
    yield
    get_graph_config.cache_clear()


def _make_fake_session(captured, create_response, statuses):
    class FakeResponse:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status = status

        def raise_for_status(self):
            if self.status >= 400:
                raise aiohttp_error(self.status)

        async def json(self):
            return self._payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeSession:
        def __init__(self):
            self._status_iter = iter(statuses)

        def post(self, url, headers=None, json=None, data=None, auth=None):
            captured["post"].append({"url": url, "headers": headers, "json": json, "data": data})
            return FakeResponse(create_response, next(self._status_iter, 200))

        def get(self, url, headers=None):
            captured["get"].append({"url": url, "headers": headers})
            return FakeResponse({"data": {"status": "running"}}, 200)

        def delete(self, url, headers=None):
            captured["delete"].append({"url": url, "headers": headers})
            return FakeResponse({}, 200)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    return FakeSession


def aiohttp_error(status):
    import aiohttp

    return aiohttp.ClientResponseError(
        request_info=None,
        history=(),
        status=status,
        message=f"HTTP {status}",
    )


def _patch_aiohttp_and_token(monkeypatch, captured, create_response, statuses=(200,)):
    async def fake_get_aura_token(cls, client_id, client_secret):
        return {"access_token": "test-token"}

    fake_session_cls = _make_fake_session(captured, create_response, statuses)

    monkeypatch.setattr(
        Neo4jAuraDevDatasetDatabaseHandler,
        "_get_aura_token",
        classmethod(fake_get_aura_token),
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: fake_session_cls())


def test_base62_encode():
    assert _base62_encode(0) == "0"
    value = 2**128
    encoded = _base62_encode(value)
    decoded = 0
    for char in encoded:
        decoded = decoded * 62 + (
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).index(char)
    assert decoded == value


def test_instance_name_for_dataset():
    with pytest.raises(ValueError, match="dataset_id is required"):
        Neo4jAuraDevDatasetDatabaseHandler._instance_name_for_dataset(None)

    direct = Neo4jAuraDevDatasetDatabaseHandler._instance_name_for_dataset(DATASET_ID)
    from_str = Neo4jAuraDevDatasetDatabaseHandler._instance_name_for_dataset(str(DATASET_ID))
    assert direct == from_str

    assert direct.startswith(NEO4J_AURA_INSTANCE_NAME_PREFIX)
    assert len(direct) <= NEO4J_AURA_INSTANCE_NAME_LIMIT

    digest = hashlib.sha256(DATASET_ID.bytes).digest()
    hash_int = int.from_bytes(digest, byteorder="big")
    expected_encoded = _base62_encode(hash_int)
    expected = (
        NEO4J_AURA_INSTANCE_NAME_PREFIX
        + expected_encoded[: NEO4J_AURA_INSTANCE_NAME_LIMIT - len(NEO4J_AURA_INSTANCE_NAME_PREFIX)]
    )
    assert direct == expected


@pytest.mark.asyncio
async def test_create_dataset_uses_default_payload(monkeypatch):
    captured = {"post": [], "get": [], "delete": []}
    create_response = {
        "data": {
            "id": "instance-1",
            "connection_url": "neo4j+s://abc.databases.neo4j.io",
            "username": "neo4j",
            "password": "super-secret",
        }
    }
    _patch_aiohttp_and_token(monkeypatch, captured, create_response)

    result = await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(DATASET_ID, None)

    payload = captured["post"][0]["json"]
    assert payload["version"] == "5"
    assert payload["region"] == "europe-west1"
    assert payload["memory"] == "1GB"
    assert payload["type"] == "professional-db"
    assert payload["cloud_provider"] == "gcp"
    assert payload["tenant_id"] == "test-tenant-id"
    assert payload["name"] == Neo4jAuraDevDatasetDatabaseHandler._instance_name_for_dataset(
        DATASET_ID
    )
    assert len(payload["name"]) <= NEO4J_AURA_INSTANCE_NAME_LIMIT

    assert result["graph_database_name"] == "neo4j"
    assert result["graph_database_url"] == "neo4j+s://abc.databases.neo4j.io"
    assert result["graph_database_provider"] == "neo4j"
    assert result["graph_dataset_database_handler"] == "neo4j_aura_dev"
    assert result["graph_database_connection_info"]["graph_database_username"] == "neo4j"
    assert result["graph_database_connection_info"]["graph_database_password"] != "super-secret"


@pytest.mark.asyncio
async def test_create_dataset_merges_allowed_kwargs_overrides(monkeypatch):
    captured = {"post": [], "get": [], "delete": []}
    create_response = {
        "data": {
            "id": "instance-1",
            "connection_url": "neo4j+s://abc.databases.neo4j.io",
            "username": "neo4j",
            "password": "super-secret",
        }
    }
    _patch_aiohttp_and_token(monkeypatch, captured, create_response)

    await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(
        DATASET_ID,
        None,
        region="us-east1",
        memory="8GB",
        type="enterprise-db",
        cloud_provider="aws",
        version="2025",
    )

    payload = captured["post"][0]["json"]
    assert payload["region"] == "us-east1"
    assert payload["memory"] == "8GB"
    assert payload["type"] == "enterprise-db"
    assert payload["cloud_provider"] == "aws"
    assert payload["version"] == "2025"
    assert payload["name"] == Neo4jAuraDevDatasetDatabaseHandler._instance_name_for_dataset(
        DATASET_ID
    )


@pytest.mark.asyncio
async def test_create_dataset_rejects_unknown_kwargs(monkeypatch):
    captured = {"post": [], "get": [], "delete": []}
    create_response = {
        "data": {
            "id": "instance-1",
            "connection_url": "neo4j+s://abc.databases.neo4j.io",
            "username": "neo4j",
            "password": "super-secret",
        }
    }
    _patch_aiohttp_and_token(monkeypatch, captured, create_response)

    with pytest.raises(ValueError, match="Unsupported Neo4j Aura payload override"):
        await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(
            DATASET_ID, None, totally_bogus="value"
        )

    assert captured["post"] == []


@pytest.mark.asyncio
async def test_create_dataset_rejects_none_kwargs(monkeypatch):
    captured = {"post": [], "get": [], "delete": []}
    create_response = {
        "data": {
            "id": "instance-1",
            "connection_url": "neo4j+s://abc.databases.neo4j.io",
            "username": "neo4j",
            "password": "super-secret",
        }
    }
    _patch_aiohttp_and_token(monkeypatch, captured, create_response)

    with pytest.raises(ValueError, match="None values are not permitted"):
        await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(DATASET_ID, None, region=None)

    assert captured["post"] == []


@pytest.mark.asyncio
async def test_create_dataset_requires_neo4j_provider(monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "ladybug")
    get_graph_config.cache_clear()

    with pytest.raises(ValueError, match="can only be used with Neo4j"):
        await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(DATASET_ID, None)


@pytest.mark.asyncio
async def test_create_dataset_requires_credentials(monkeypatch):
    monkeypatch.delenv("NEO4J_CLIENT_ID", raising=False)

    with pytest.raises(ValueError, match="environment variables must be set"):
        await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(DATASET_ID, None)


@pytest.mark.asyncio
async def test_create_dataset_polls_until_instance_running(monkeypatch):
    captured = {"post": [], "get": [], "delete": []}
    create_response = {
        "data": {
            "id": "instance-1",
            "connection_url": "neo4j+s://abc.databases.neo4j.io",
            "username": "neo4j",
            "password": "super-secret",
        }
    }

    class FakeResponse:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status = status

        def raise_for_status(self):
            if self.status >= 400:
                raise aiohttp_error(self.status)

        async def json(self):
            return self._payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    poll_calls = {"n": 0}

    class FakeSession:
        def post(self, url, headers=None, json=None, data=None, auth=None):
            captured["post"].append({"url": url, "json": json})
            return FakeResponse(create_response, 200)

        def get(self, url, headers=None):
            captured["get"].append({"url": url})
            poll_calls["n"] += 1
            status = "running" if poll_calls["n"] >= 3 else "creating"
            return FakeResponse({"data": {"status": status}}, 200)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_get_aura_token(cls, client_id, client_secret):
        return {"access_token": "test-token"}

    monkeypatch.setattr(
        Neo4jAuraDevDatasetDatabaseHandler,
        "_get_aura_token",
        classmethod(fake_get_aura_token),
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: FakeSession())

    original_asyncio_sleep = asyncio.sleep

    async def fast_sleep(*args, **kwargs):
        await original_asyncio_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(DATASET_ID, None)

    assert len(captured["post"]) == 1
    assert len(captured["get"]) == 3


@pytest.mark.asyncio
async def test_create_then_resolve_round_trips_password(monkeypatch):
    captured = {"post": [], "get": [], "delete": []}
    create_response = {
        "data": {
            "id": "instance-1",
            "connection_url": "neo4j+s://abc.databases.neo4j.io",
            "username": "neo4j",
            "password": "super-secret",
        }
    }
    _patch_aiohttp_and_token(monkeypatch, captured, create_response)

    info = await Neo4jAuraDevDatasetDatabaseHandler.create_dataset(DATASET_ID, None)

    dataset_database = SimpleNamespace(
        graph_database_connection_info=dict(info["graph_database_connection_info"])
    )
    resolved = await Neo4jAuraDevDatasetDatabaseHandler.resolve_dataset_connection_info(
        dataset_database
    )

    assert resolved.graph_database_connection_info["graph_database_password"] == "super-secret"
