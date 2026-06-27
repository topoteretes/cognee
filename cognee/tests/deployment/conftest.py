"""Pytest fixtures for deployment end-to-end tests."""

from __future__ import annotations

import os
import subprocess
import uuid

import httpx
import pytest
import pytest_asyncio

from cognee.tests.deployment.golden_flow import run_golden_flow
from cognee.tests.deployment.helpers import (
    build_api_container_cmd,
    build_mcp_container_cmd,
    get_free_port,
    resolve_docker_image,
    stop_container,
    stream_container_logs,
    wait_for_health,
)
from cognee.tests.deployment.mock_llm import start_mock_llm_server, stop_mock_llm_server

pytestmark = pytest.mark.deployment


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--image",
        action="store",
        default=None,
        help="Docker image for API deployment tests: build, pull, or an image name",
    )
    parser.addoption(
        "--mcp-image",
        action="store",
        default=None,
        help="Docker image for MCP deployment tests: build, pull, or an image name",
    )


def _use_real_llm() -> bool:
    return os.getenv("DEPLOYMENT_USE_REAL_LLM", "").lower() in ("true", "1", "yes")


@pytest.fixture(scope="session")
def mock_llm_server():
    if _use_real_llm():
        yield None
        return

    server, _thread, port = start_mock_llm_server()
    url = f"http://127.0.0.1:{port}/v1"
    yield url
    stop_mock_llm_server(server)


@pytest.fixture(scope="session")
def image_name(request):
    return resolve_docker_image(
        image_option=request.config.getoption("--image"),
        env_var="COGNEE_DOCKER_IMAGE",
        local_tag="cognee:local",
        published_image="cognee/cognee:latest",
        build_cmd=["docker", "build", "-t", "cognee:local", "."],
    )


@pytest.fixture(scope="session")
def mcp_image_name(request):
    return resolve_docker_image(
        image_option=request.config.getoption("--mcp-image"),
        env_var="COGNEE_MCP_IMAGE",
        local_tag="cognee-mcp:local",
        published_image="cognee/cognee-mcp:latest",
        build_cmd=[
            "docker",
            "build",
            "-t",
            "cognee-mcp:local",
            "-f",
            "cognee-mcp/Dockerfile",
            ".",
        ],
    )


@pytest.fixture
def running_container(request, mock_llm_server, image_name):
    if _use_real_llm() and mock_llm_server is None:
        pytest.skip("Real LLM mode requires DEPLOYMENT_USE_REAL_LLM secrets to be configured.")

    host_port = get_free_port()
    container_name = f"cognee-test-{uuid.uuid4().hex[:8]}"
    cmd = build_api_container_cmd(
        image_name=image_name,
        container_name=container_name,
        host_port=host_port,
        mock_llm_url=mock_llm_server or "",
    )

    print(f"Starting container {container_name} on port {host_port} with image {image_name}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed to start: {proc.stderr}")

    url = f"http://127.0.0.1:{host_port}"
    failed = False

    try:
        wait_for_health(f"{url}/health", timeout=120.0)
        yield {"url": url, "port": host_port, "container_name": container_name}
    except Exception:
        failed = True
        raise
    finally:
        test_failed = getattr(getattr(request.node, "rep_call", None), "failed", False)
        if failed or test_failed:
            print(f"\n--- CONTAINER LOGS FOR {container_name} ---")
            stream_container_logs(container_name)
        stop_container(container_name)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest_asyncio.fixture
async def api_client(running_container):
    async with httpx.AsyncClient(base_url=running_container["url"], timeout=60.0) as client:
        yield client


@pytest_asyncio.fixture
async def mcp_client(mcp_image_name):
    host_port = get_free_port()
    container_name = f"cognee-mcp-test-{uuid.uuid4().hex[:8]}"
    cmd = build_mcp_container_cmd(
        image_name=mcp_image_name,
        container_name=container_name,
        host_port=host_port,
    )

    print(f"Starting MCP container {container_name} on port {host_port}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"docker run for MCP failed to start: {proc.stderr}")

    url = f"http://127.0.0.1:{host_port}"

    class MCPAsyncClient(httpx.AsyncClient):
        async def call_tool(self, tool_name: str, arguments: dict) -> dict:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            resp = await self.post("/mcp", json=payload)
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                raise RuntimeError(f"MCP tool error: {result['error']}")
            return result["result"]

    failed = False
    try:
        wait_for_health(f"{url}/health", timeout=120.0)
        async with MCPAsyncClient(base_url=url, timeout=60.0) as client:
            yield client
    except Exception:
        failed = True
        raise
    finally:
        if failed:
            print(f"\n--- MCP CONTAINER LOGS FOR {container_name} ---")
            stream_container_logs(container_name)
        stop_container(container_name)


@pytest.fixture
def golden_flow():
    return run_golden_flow
