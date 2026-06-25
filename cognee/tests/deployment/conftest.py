"""Shared fixtures for deployment smoke tests.

Provides Docker image build and container lifecycle helpers. Teardown
is unconditional so containers are always removed, even on failure.
"""

import subprocess
import time
from typing import Generator

import httpx
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "deployment: deployment/container smoke tests")


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def wait_for_health(url: str, timeout: int = 120, interval: float = 2.0):
    """Poll url until it returns HTTP 2xx or timeout seconds elapse."""
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code < 300:
                return resp
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_err = exc
        time.sleep(interval)
    raise TimeoutError(f"{url} not healthy after {timeout}s (last: {last_err})")


def _dump_logs_and_remove(name: str):
    logs = _docker("logs", "--tail", "80", name, check=False)
    if logs.stdout:
        print(f"\n--- {name} stdout ---\n{logs.stdout[-3000:]}")
    if logs.stderr:
        print(f"\n--- {name} stderr ---\n{logs.stderr[-3000:]}")
    _docker("rm", "-f", name, check=False)


@pytest.fixture(scope="session")
def docker_network() -> Generator[str, None, None]:
    name = "cognee-mcp-e2e-net"
    _docker("network", "create", name, check=False)
    yield name
    _docker("network", "rm", name, check=False)


@pytest.fixture(scope="session")
def backend_image() -> str:
    tag = "cognee-backend:mcp-e2e"
    result = _docker("build", "-t", tag, "-f", "Dockerfile", ".", check=False)
    if result.returncode != 0:
        pytest.skip(f"Backend image build failed:\n{result.stderr[-500:]}")
    return tag


@pytest.fixture(scope="session")
def mcp_image() -> str:
    tag = "cognee-mcp:e2e"
    result = _docker("build", "-t", tag, "-f", "cognee-mcp/Dockerfile", ".", check=False)
    if result.returncode != 0:
        pytest.skip(f"MCP image build failed:\n{result.stderr[-500:]}")
    return tag


@pytest.fixture(scope="session")
def backend_container(backend_image: str, docker_network: str) -> Generator[str, None, None]:
    name = "cognee-mcp-e2e-backend"
    _docker("rm", "-f", name, check=False)

    _docker(
        "run",
        "-d",
        "--name",
        name,
        "--network",
        docker_network,
        "-p",
        "18000:8000",
        "-e",
        "ENV=dev",
        "-e",
        "LLM_API_KEY=test-key",
        "-e",
        "ENABLE_BACKEND_ACCESS_CONTROL=false",
        backend_image,
    )

    try:
        wait_for_health("http://localhost:18000/health", timeout=120)
        yield name
    finally:
        _dump_logs_and_remove(name)


@pytest.fixture(scope="session")
def mcp_container(
    mcp_image: str, docker_network: str, backend_container: str
) -> Generator[str, None, None]:
    """Start the MCP server in API mode, pointed at the backend container."""
    name = "cognee-mcp-e2e-server"
    _docker("rm", "-f", name, check=False)

    _docker(
        "run",
        "-d",
        "--name",
        name,
        "--network",
        docker_network,
        "-p",
        "18001:8000",
        "-e",
        "TRANSPORT_MODE=http",
        "-e",
        f"API_URL=http://{backend_container}:8000",
        "-e",
        "MCP_DISABLE_DNS_REBINDING_PROTECTION=true",
        "-e",
        "LLM_API_KEY=test-key",
        mcp_image,
    )

    try:
        wait_for_health("http://localhost:18001/health", timeout=90)
        yield name
    finally:
        _dump_logs_and_remove(name)


@pytest.fixture(scope="session")
def mcp_url(mcp_container: str) -> str:
    return "http://localhost:18001"


@pytest.fixture(scope="session")
def backend_url(backend_container: str) -> str:
    return "http://localhost:18000"
