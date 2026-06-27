"""Shared helpers for deployment end-to-end tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def wait_for_health(url: str, timeout: float = 60.0) -> None:
    """Poll until the service health endpoint reports ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") in ("ready", "ok") or data.get("health") == "healthy":
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Service at {url} not ready after {timeout}s")


def resolve_docker_image(
    image_option: str | None,
    env_var: str,
    local_tag: str,
    published_image: str,
    build_cmd: list[str],
) -> str:
    image = image_option or os.environ.get(env_var)
    if not image:
        image = "build"

    if image == "build":
        print(f"Building local Docker image '{local_tag}'...")
        proc = subprocess.run(build_cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode == 0:
            print(f"Successfully built local image '{local_tag}'.")
            return local_tag
        print(f"Failed to build local Docker image: {proc.stderr}")
        print(f"Falling back to published image '{published_image}'.")
        return published_image

    if image == "pull":
        return published_image

    return image


def mock_llm_endpoint_for_container(mock_llm_url: str) -> str:
    parsed = urlparse(mock_llm_url)
    mock_llm_port = parsed.port
    if sys.platform.startswith("linux"):
        return f"http://127.0.0.1:{mock_llm_port}/v1"
    return f"http://host.docker.internal:{mock_llm_port}/v1"


def build_api_container_cmd(
    *,
    image_name: str,
    container_name: str,
    host_port: int,
    mock_llm_url: str,
) -> list[str]:
    llm_endpoint = mock_llm_endpoint_for_container(mock_llm_url)
    env = [
        "ENV=local",
        "DB_PROVIDER=sqlite",
        "COGNEE_SKIP_CONNECTION_TEST=true",
        "LLM_PROVIDER=openai",
        "LLM_MODEL=gpt-5-mini",
        f"LLM_ENDPOINT={llm_endpoint}",
        "LLM_API_KEY=mock-key",
        "EMBEDDING_PROVIDER=openai",
        "EMBEDDING_MODEL=text-embedding-3-small",
        "EMBEDDING_DIMENSIONS=1536",
        f"EMBEDDING_ENDPOINT={llm_endpoint}",
        "EMBEDDING_API_KEY=mock-key",
    ]

    cmd = ["docker", "run", "-d", "--name", container_name]
    if sys.platform.startswith("linux"):
        cmd.extend(["--network", "host", "-e", f"HTTP_PORT={host_port}"])
    else:
        cmd.extend(
            [
                "-p",
                f"{host_port}:8000",
                "--add-host",
                "host.docker.internal:host-gateway",
            ]
        )

    for entry in env:
        cmd.extend(["-e", entry])

    cmd.append(image_name)
    return cmd


def fetch_container_logs(container_name: str) -> tuple[str, str]:
    proc = subprocess.run(
        ["docker", "logs", container_name],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return proc.stdout, proc.stderr


def stream_container_logs(container_name: str) -> None:
    stdout, stderr = fetch_container_logs(container_name)
    sys.stdout.buffer.write(stdout.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stderr.buffer.write(stderr.encode("utf-8", errors="replace"))
    sys.stderr.buffer.write(b"\n")
    sys.stdout.flush()
    sys.stderr.flush()


def stop_container(container_name: str) -> None:
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)


def assert_no_tracebacks_in_logs(stdout: str, stderr: str) -> None:
    assert "Traceback" not in stdout, f"Traceback found in container stdout logs:\n{stdout}"
    assert "Traceback" not in stderr, f"Traceback found in container stderr logs:\n{stderr}"


def build_mcp_container_cmd(*, image_name: str, container_name: str, host_port: int) -> list[str]:
    cmd = ["docker", "run", "-d", "--name", container_name]
    if sys.platform.startswith("linux"):
        cmd.extend(
            ["--network", "host", "-e", f"HTTP_PORT={host_port}", "-e", "TRANSPORT_MODE=http"]
        )
    else:
        cmd.extend(["-p", f"{host_port}:8000", "-e", "TRANSPORT_MODE=http"])
    cmd.append(image_name)
    return cmd
