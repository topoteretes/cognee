"""Helpers for the MCP Docker HTTP e2e suite (T3 / #3361).

Drive the built ``cognee-mcp`` image over HTTP (Docker CLI + httpx + mcp): image
availability/build, a free-port picker, a health poller, a container context
manager with guaranteed teardown, and an MCP streamable-HTTP client session.
"""

from __future__ import annotations

import contextlib
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

# mcp_harness.py -> deployment -> tests -> cognee -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

# Port the MCP server listens on inside the container (entrypoint default).
CONTAINER_HTTP_PORT = 8000


def docker_available() -> bool:
    """Return True when a usable Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def image_exists(tag: str) -> bool:
    """Return True when ``tag`` is present in the local image store."""
    return (
        subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
        ).returncode
        == 0
    )


def ensure_mcp_image(tag: str) -> str:
    """Return ``tag`` if present locally, otherwise build it from cognee-mcp/Dockerfile."""
    if image_exists(tag):
        return tag

    dockerfile = REPO_ROOT / "cognee-mcp" / "Dockerfile"
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(dockerfile),
            "-t",
            tag,
            str(REPO_ROOT),
        ],
        check=True,
    )
    return tag


def free_port() -> int:
    """Reserve and return a free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_health(url: str, timeout: float = 120.0) -> None:
    """Poll an HTTP health endpoint until it returns 200 or raise ``TimeoutError``."""
    import httpx

    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 - poller intentionally tolerant
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(
        f"Health check {url!r} did not pass within {timeout:.0f}s; last error: {last_error}"
    )


class MCPContainer:
    """A running ``cognee-mcp`` container in HTTP transport (direct) mode."""

    def __init__(self, name: str, host_port: int):
        self.name = name
        self.host_port = host_port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp"

    def logs(self) -> str:
        """Return combined stdout+stderr from ``docker logs`` (for debugging)."""
        result = subprocess.run(
            ["docker", "logs", self.name],
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr


@contextlib.contextmanager
def run_mcp_http_container(
    image: str,
    *,
    extra_env: Optional[dict] = None,
    name_suffix: str = "",
    health_timeout: float = 120.0,
) -> Iterator[MCPContainer]:
    """Start cognee-mcp with ``TRANSPORT_MODE=http``, yield it, always tear down.

    ``LLM_API_KEY`` is a dummy: the session-cache remember/recall path never calls
    a model, so the run is deterministic and key-free.
    """
    port = free_port()
    name = f"cognee-mcp-t3-{port}{name_suffix}"

    env = {
        "TRANSPORT_MODE": "http",
        "LLM_API_KEY": "mock-key",
        "ENV": "local",
    }
    if extra_env:
        env.update(extra_env)

    env_args: list[str] = []
    for key, value in env.items():
        env_args += ["-e", f"{key}={value}"]

    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"{port}:{CONTAINER_HTTP_PORT}",
            *env_args,
            image,
        ],
        check=True,
    )

    container = MCPContainer(name=name, host_port=port)
    try:
        try:
            wait_for_health(container.health_url, timeout=health_timeout)
        except Exception:
            print(
                f"\n----- docker logs {name} -----\n"
                f"{container.logs()}\n"
                f"------------------------------"
            )
            raise
        yield container
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@contextlib.asynccontextmanager
async def mcp_client_session(mcp_url: str) -> AsyncIterator[object]:
    """Open an initialized MCP client session over streamable HTTP at ``/mcp``."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(mcp_url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session
