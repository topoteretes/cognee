import os
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime, timezone

import httpx
import pytest


COMPOSE_FILE = os.getenv("COGNEE_E2E_COMPOSE_FILE", "docker-compose.yml")
COMPOSE_PROFILES = ("postgres", "mcp")
API_BASE_URL = os.getenv("COGNEE_E2E_API_URL", "http://localhost:8000")
MCP_BASE_URL = os.getenv("COGNEE_E2E_MCP_URL", "http://localhost:8001")


def compose_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", "-f", COMPOSE_FILE]
    for profile in COMPOSE_PROFILES:
        command.extend(["--profile", profile])
    command.extend(args)

    return subprocess.run(command, check=check, capture_output=True, text=True)


def wait_for_http(url: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(url)
                if response.status_code < 500:
                    return response.json()
            except (httpx.HTTPError, ValueError) as error:
                last_error = error

            time.sleep(2)

    raise AssertionError(f"{url} did not become ready within {timeout}s: {last_error}")


def assert_no_log_tracebacks(
    services: Iterable[str] = ("cognee", "cognee-mcp", "postgres"),
    since: str | None = None,
) -> None:
    log_args = ["logs", "--no-color"]
    if since is not None:
        log_args.extend(["--since", since])
    log_args.extend(services)

    result = compose_command(*log_args)
    forbidden_markers = ("Traceback (most recent call last)",)
    found_markers = [marker for marker in forbidden_markers if marker in result.stdout]
    assert not found_markers, f"Unexpected traceback marker(s) in compose logs: {found_markers}"


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return API_BASE_URL


@pytest.fixture(scope="session")
def mcp_base_url() -> str:
    return MCP_BASE_URL


@pytest.fixture(scope="session", autouse=True)
def services_are_ready(api_base_url: str, mcp_base_url: str):
    wait_for_http(f"{api_base_url}/health")
    wait_for_http(f"{mcp_base_url}/health")
    test_started_at = datetime.now(timezone.utc).isoformat()
    yield
    assert_no_log_tracebacks(since=test_started_at)
