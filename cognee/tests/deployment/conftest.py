"""Shared fixtures for deployment smoke tests.

Provides Docker image build + container lifecycle helpers that future
deployment tests (T0 harness) can absorb. Keeps teardown unconditional
so containers are always removed, even on failure.
"""

import subprocess
import time
from typing import Generator

import httpx
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "deployment: deployment/container smoke tests")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["docker", *args], check=check)


def wait_for_health(url: str, timeout: int = 120, interval: float = 2.0):
    """Poll *url* until it returns HTTP 2xx or *timeout* seconds elapse."""
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


@pytest.fixture(scope="session")
def backend_image() -> str:
    """Build the backend Docker image and return the tag."""
    tag = "cognee-backend:smoke-test"
    result = _docker(
        "build",
        "-t",
        tag,
        "-f",
        "Dockerfile",
        ".",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"Backend image build failed:\n{result.stderr[-500:]}")
    return tag


@pytest.fixture(scope="session")
def frontend_image() -> str:
    """Build the frontend Docker image and return the tag."""
    tag = "cognee-frontend:smoke-test"
    result = _docker(
        "build",
        "-t",
        tag,
        "-f",
        "cognee-frontend/Dockerfile",
        "cognee-frontend/",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"Frontend image build failed:\n{result.stderr[-500:]}")
    return tag


@pytest.fixture(scope="session")
def docker_network() -> Generator[str, None, None]:
    """Create an isolated Docker network for the test session."""
    name = "cognee-smoke-test-net"
    _docker("network", "create", name, check=False)
    yield name
    _docker("network", "rm", name, check=False)


@pytest.fixture(scope="session")
def backend_container(backend_image: str, docker_network: str) -> Generator[str, None, None]:
    """Run the backend container and wait for /health to respond."""
    name = "cognee-smoke-backend"
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
        logs = _docker("logs", "--tail", "50", name, check=False)
        if logs.stdout:
            print(f"\n--- backend logs ---\n{logs.stdout[-2000:]}")
        if logs.stderr:
            print(f"\n--- backend stderr ---\n{logs.stderr[-2000:]}")
        _docker("rm", "-f", name, check=False)


@pytest.fixture(scope="session")
def frontend_container(
    frontend_image: str, docker_network: str, backend_container: str
) -> Generator[str, None, None]:
    """Run the frontend container pointed at the backend."""
    name = "cognee-smoke-frontend"
    _docker("rm", "-f", name, check=False)

    _docker(
        "run",
        "-d",
        "--name",
        name,
        "--network",
        docker_network,
        "-p",
        "13000:3000",
        "-e",
        f"NEXT_PUBLIC_BACKEND_API_URL=http://{backend_container}:8000",
        frontend_image,
    )

    try:
        wait_for_health("http://localhost:13000", timeout=90)
        yield name
    finally:
        logs = _docker("logs", "--tail", "50", name, check=False)
        if logs.stdout:
            print(f"\n--- frontend logs ---\n{logs.stdout[-2000:]}")
        if logs.stderr:
            print(f"\n--- frontend stderr ---\n{logs.stderr[-2000:]}")
        _docker("rm", "-f", name, check=False)
