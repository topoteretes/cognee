import os
import pytest
import pytest_asyncio
import asyncio
from fastapi.testclient import TestClient

import cognee
from cognee.api.client import app
from cognee.modules.users.methods import get_default_user, get_authenticated_user


async def _reset_engines_and_prune():
    """Reset db engine caches and prune data/system."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

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
    yield
    os.environ.clear()
    os.environ.update(original_env)


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


@pytest.mark.asyncio
async def test_mcp_tool_logging(e2e_config, cache_engine):
    """Test that MCP tools succeed and log to Redis."""
    import sys
    import importlib.util
    from pathlib import Path

    await _reset_engines_and_prune()

    repo_root = Path(__file__).parent.parent.parent
    mcp_src_path = repo_root / "cognee-mcp" / "src"
    mcp_server_path = mcp_src_path / "server.py"

    if not mcp_server_path.exists():
        pytest.skip(f"MCP server not found at {mcp_server_path}")

    if str(mcp_src_path) not in sys.path:
        sys.path.insert(0, str(mcp_src_path))

    spec = importlib.util.spec_from_file_location("mcp_server_module", mcp_server_path)
    mcp_server_module = importlib.util.module_from_spec(spec)

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(str(mcp_src_path))
        spec.loader.exec_module(mcp_server_module)
    finally:
        os.chdir(original_cwd)

    if mcp_server_module.cognee_client is None:
        cognee_client_path = mcp_src_path / "cognee_client.py"
        if cognee_client_path.exists():
            spec_client = importlib.util.spec_from_file_location(
                "cognee_client", cognee_client_path
            )
            cognee_client_module = importlib.util.module_from_spec(spec_client)
            spec_client.loader.exec_module(cognee_client_module)
            CogneeClient = cognee_client_module.CogneeClient
            mcp_server_module.cognee_client = CogneeClient()
        else:
            pytest.skip(f"CogneeClient not found at {cognee_client_path}")

    test_text = "Germany is located in Europe right next to the Netherlands."
    await mcp_server_module.cognify(data=test_text)
    await asyncio.sleep(30.0)

    list_result = await mcp_server_module.list_data()
    assert list_result is not None, "List data should return results"

    search_result = await mcp_server_module.search(
        search_query="Germany", search_type="GRAPH_COMPLETION", top_k=5
    )
    assert search_result is not None, "Search should return results"

    interaction_data = "User: What is Germany?\nAgent: Germany is a country in Europe."
    await mcp_server_module.save_interaction(data=interaction_data)
    await asyncio.sleep(30.0)

    status_result = await mcp_server_module.cognify_status()
    assert status_result is not None, "Cognify status should return results"

    await mcp_server_module.prune()
    await asyncio.sleep(0.5)

    logs = await cache_engine.get_usage_logs("unknown", limit=50)
    mcp_logs = [log for log in logs if log.get("type") == "mcp_tool"]
    assert len(mcp_logs) > 0, (
        f"Should have MCP tool logs with user_id='unknown'. Found logs: {[log.get('function_name') for log in logs[:5]]}"
    )
    assert len(mcp_logs) == 6
    function_names = [log.get("function_name") for log in mcp_logs]
    expected_tools = [
        "MCP cognify",
        "MCP list_data",
        "MCP search",
        "MCP save_interaction",
        "MCP cognify_status",
        "MCP prune",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in function_names, (
            f"Should have {expected_tool} log. Found: {function_names}"
        )

    for log in mcp_logs:
        assert log["type"] == "mcp_tool"
        assert log["user_id"] == "unknown"
        assert log["success"] is True
