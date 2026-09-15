import asyncio
import logging
import os

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import cognee
from cognee.api.client import app
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
from cognee.modules.users.methods import get_authenticated_user, get_default_user

logger = logging.getLogger(__name__)


async def _reset_engines_and_prune():
    """Reset db engine caches and prune data/system."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine_async

        vector_engine = await get_vector_engine_async()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        logger.debug("Ignoring exception in _reset_engines_and_prune", exc_info=True)

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
    os.environ.setdefault("CACHE_HOST", "localhost")
    os.environ.setdefault("CACHE_PORT", "6379")
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
    from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

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


@pytest.mark.asyncio
async def test_mcp_tool_logging(e2e_config, mcp_data_setup, cache_engine):
    """Live MCP tools log to Redis when driven over the MCP protocol.

    Goes through ``fastmcp.Client`` rather than reaching for module attributes:
    only registered tools are reachable by a real client, and it is
    ``@registry.tool`` -- not a hand-written decorator -- that folds in
    ``@log_usage`` (cognee-mcp/src/tool_registry.py). Calling the functions
    directly, as this test used to, exercised neither.
    """
    import sys
    from pathlib import Path

    mcp_root = Path(__file__).resolve().parents[2] / "cognee-mcp"
    if not (mcp_root / "src" / "server.py").exists():
        pytest.skip(f"MCP server not found at {mcp_root}")

    fastmcp = pytest.importorskip("fastmcp")

    if str(mcp_root) not in sys.path:
        sys.path.insert(0, str(mcp_root))

    from src import server as mcp_server
    from src.cognee_client import CogneeClient

    if mcp_server.cognee_client is None:
        # No api_url => use_api False => the in-process SDK path.
        mcp_server.cognee_client = CogneeClient()

    async with fastmcp.Client(mcp_server.mcp) as client:
        # `recall` takes `datasets` as a comma-separated string; `cognify_status`
        # takes a single `dataset_name`.
        await client.call_tool("recall", {"query": "Germany", "datasets": mcp_data_setup})
        await client.call_tool("cognify_status", {"dataset_name": mcp_data_setup})

    logs = await cache_engine.get_usage_logs("unknown", limit=50)
    mcp_logs = {log.get("function_name"): log for log in logs if log.get("type") == "mcp_tool"}

    for name in ("MCP recall", "MCP cognify_status"):
        assert name in mcp_logs, f"Missing {name} usage log. Found: {sorted(mcp_logs)}"
        assert mcp_logs[name]["type"] == "mcp_tool"
        assert mcp_logs[name]["success"] is True
