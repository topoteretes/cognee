import os
import pytest
import pytest_asyncio
import asyncio
from fastapi.testclient import TestClient

import cognee
from cognee.api.client import app
from cognee.modules.users.methods import get_default_user, get_authenticated_user
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)


async def _reset_engines_and_prune():
    """Reset db engine caches and prune data/system."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

    create_graph_engine.cache_clear()
    create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.fixture(scope="session")
def event_loop():
    """Use a single asyncio event loop for this test module."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture(scope="session")
def e2e_config():
    """Configure environment for E2E tests."""
    original_env = os.environ.copy()
    os.environ["USAGE_LOGGING"] = "true"
    os.environ["CACHE_BACKEND"] = "redis"
    os.environ["CACHE_HOST"] = "localhost"
    os.environ["CACHE_PORT"] = "6379"
    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)
    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()


@pytest.fixture(scope="session")
def authenticated_client(test_client):
    """Override authentication to use default user."""

    async def override_get_authenticated_user():
        return await get_default_user()

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


@pytest_asyncio.fixture(scope="session")
async def test_data_setup():
    """Set up test data: prune first, then add file and cognify."""
    await _reset_engines_and_prune()

    dataset_name = "test_e2e_dataset"
    test_text = "Germany is located in Europe right next to the Netherlands."

    await cognee.add(test_text, dataset_name)
    await cognee.cognify([dataset_name])

    yield dataset_name

    await _reset_engines_and_prune()


@pytest_asyncio.fixture
async def mcp_data_setup():
    """Set up test data for MCP tests: prune first, then add file and cognify."""
    await _reset_engines_and_prune()

    dataset_name = "test_mcp_dataset"
    test_text = "Germany is located in Europe right next to the Netherlands."

    await cognee.add(test_text, dataset_name)
    await cognee.cognify([dataset_name])

    yield dataset_name

    await _reset_engines_and_prune()


@pytest.fixture(scope="session")
def test_client():
    """TestClient instance for API calls."""
    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def cache_engine(e2e_config):
    """Get cache engine for log verification in test's event loop."""
    create_cache_engine.cache_clear()
    from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter
    from cognee.infrastructure.databases.cache.config import get_cache_config

    config = get_cache_config()
    if not config.usage_logging or config.cache_backend != "redis":
        pytest.skip("Redis usage logging not configured")

    engine = RedisAdapter(
        host=config.cache_host,
        port=config.cache_port,
        username=config.cache_username,
        password=config.cache_password,
        log_key="usage_logs",
    )
    return engine


@pytest.mark.asyncio
async def test_api_endpoint_logging(e2e_config, authenticated_client, cache_engine):
    """Test that API endpoints succeed and log to Redis."""
    user = await get_default_user()
    dataset_name = "test_e2e_api_dataset"

    add_response = authenticated_client.post(
        "/api/v1/add",
        data={"datasetName": dataset_name},
        files=[
            (
                "data",
                (
                    "test.txt",
                    b"Germany is located in Europe right next to the Netherlands.",
                    "text/plain",
                ),
            )
        ],
    )
    assert add_response.status_code in [200, 201], f"Add endpoint failed: {add_response.text}"

    cognify_response = authenticated_client.post(
        "/api/v1/cognify",
        json={"datasets": [dataset_name], "run_in_background": False},
    )
    assert cognify_response.status_code in [200, 201], (
        f"Cognify endpoint failed: {cognify_response.text}"
    )

    search_response = authenticated_client.post(
        "/api/v1/search",
        json={"query": "Germany", "search_type": "GRAPH_COMPLETION", "datasets": [dataset_name]},
    )
    assert search_response.status_code == 200, f"Search endpoint failed: {search_response.text}"

    logs = await cache_engine.get_usage_logs(str(user.id), limit=20)

    add_logs = [log for log in logs if log.get("function_name") == "POST /v1/add"]
    assert len(add_logs) > 0
    assert add_logs[0]["type"] == "api_endpoint"
    assert add_logs[0]["user_id"] == str(user.id)
    assert add_logs[0]["success"] is True

    cognify_logs = [log for log in logs if log.get("function_name") == "POST /v1/cognify"]
    assert len(cognify_logs) > 0
    assert cognify_logs[0]["type"] == "api_endpoint"
    assert cognify_logs[0]["user_id"] == str(user.id)
    assert cognify_logs[0]["success"] is True

    search_logs = [log for log in logs if log.get("function_name") == "POST /v1/search"]
    assert len(search_logs) > 0
    assert search_logs[0]["type"] == "api_endpoint"
    assert search_logs[0]["user_id"] == str(user.id)
    assert search_logs[0]["success"] is True
