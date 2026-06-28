"""Shared helpers for deployment end-to-end tests.

Adapted from the deployment harness proposed in PR #3563, extended for the
backend-DB matrix (T10, issue #3368): per-stack database container bring-up
and per-backend environment profiles.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

import httpx

# Database container images and default credentials used by the matrix.
POSTGRES_IMAGE = "pgvector/pgvector:pg17"
NEO4J_IMAGE = "neo4j:5.26"
POSTGRES_USER = "cognee"
POSTGRES_PASSWORD = "cognee"
POSTGRES_DB = "cognee_db"
NEO4J_PASSWORD = "pleaseletmein"


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


def db_host_for_container() -> str:
    """Host alias an API container uses to reach a DB published on the host.

    On Linux the API container runs with ``--network host``, so a DB whose port
    is published to the host is reachable on ``127.0.0.1``. On macOS/Windows
    Docker Desktop the container reaches host-published ports via
    ``host.docker.internal``.
    """
    if sys.platform.startswith("linux"):
        return "127.0.0.1"
    return "host.docker.internal"


# --- Database container bring-up ------------------------------------------


def _run_or_raise(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed: {proc.stderr or proc.stdout}")
    return proc


def start_postgres_container() -> tuple[str, int]:
    """Start a pgvector-enabled Postgres and wait until it accepts connections.

    Returns ``(container_name, host_port)`` where ``host_port`` maps to the
    container's 5432.
    """
    name = f"cognee-pg-{uuid.uuid4().hex[:8]}"
    host_port = get_free_port()
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"POSTGRES_USER={POSTGRES_USER}",
        "-e",
        f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
        "-e",
        f"POSTGRES_DB={POSTGRES_DB}",
        "-p",
        f"{host_port}:5432",
        POSTGRES_IMAGE,
    ]
    _run_or_raise(cmd, f"docker run postgres ({name})")
    _wait_for_postgres(name)
    return name, host_port


def _wait_for_postgres(container_name: str, timeout: float = 90.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        proc = subprocess.run(
            ["docker", "exec", container_name, "pg_isready", "-U", POSTGRES_USER],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode == 0:
            return
        time.sleep(1)
    raise TimeoutError(f"Postgres container {container_name} not ready after {timeout}s")


def start_neo4j_container() -> tuple[str, int]:
    """Start Neo4j and wait until the bolt endpoint accepts queries.

    Returns ``(container_name, bolt_host_port)`` where ``bolt_host_port`` maps
    to the container's 7687.
    """
    name = f"cognee-neo4j-{uuid.uuid4().hex[:8]}"
    bolt_port = get_free_port()
    http_port = get_free_port()
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"NEO4J_AUTH=neo4j/{NEO4J_PASSWORD}",
        "-p",
        f"{bolt_port}:7687",
        "-p",
        f"{http_port}:7474",
        NEO4J_IMAGE,
    ]
    _run_or_raise(cmd, f"docker run neo4j ({name})")
    _wait_for_neo4j(name)
    return name, bolt_port


def _wait_for_neo4j(container_name: str, timeout: float = 120.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "cypher-shell",
                "-u",
                "neo4j",
                "-p",
                NEO4J_PASSWORD,
                "RETURN 1;",
            ],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"Neo4j container {container_name} not ready after {timeout}s")


# --- Per-stack environment profiles ---------------------------------------


def _postgres_relational_env(pg_host: str | None, pg_port: int | None) -> list[str]:
    if not pg_host or not pg_port:
        raise ValueError("Postgres host/port are required for this stack")
    return [
        "DB_PROVIDER=postgres",
        f"DB_HOST={pg_host}",
        f"DB_PORT={pg_port}",
        f"DB_USERNAME={POSTGRES_USER}",
        f"DB_PASSWORD={POSTGRES_PASSWORD}",
        f"DB_NAME={POSTGRES_DB}",
    ]


def db_env_for_stack(
    stack: str,
    *,
    pg_host: str | None = None,
    pg_port: int | None = None,
    neo4j_host: str | None = None,
    neo4j_bolt_port: int | None = None,
) -> list[str]:
    """Return the ``KEY=VALUE`` backend env entries for a matrix stack.

    Credentials are passed explicitly for every backend so the stacks behave
    identically whether ``ENABLE_BACKEND_ACCESS_CONTROL`` is on or off (pgvector
    and postgres-graph only fall back to the relational ``DB_*`` vars when access
    control is disabled).
    """
    if stack == "sqlite_lancedb_kuzu":
        return [
            "DB_PROVIDER=sqlite",
            "VECTOR_DB_PROVIDER=lancedb",
            "GRAPH_DATABASE_PROVIDER=kuzu",
        ]

    if stack == "postgres_pgvector_postgresgraph":
        return _postgres_relational_env(pg_host, pg_port) + [
            "VECTOR_DB_PROVIDER=pgvector",
            f"VECTOR_DB_HOST={pg_host}",
            f"VECTOR_DB_PORT={pg_port}",
            f"VECTOR_DB_USERNAME={POSTGRES_USER}",
            f"VECTOR_DB_PASSWORD={POSTGRES_PASSWORD}",
            f"VECTOR_DB_NAME={POSTGRES_DB}",
            "GRAPH_DATABASE_PROVIDER=postgres",
            f"GRAPH_DATABASE_HOST={pg_host}",
            f"GRAPH_DATABASE_PORT={pg_port}",
            f"GRAPH_DATABASE_USERNAME={POSTGRES_USER}",
            f"GRAPH_DATABASE_PASSWORD={POSTGRES_PASSWORD}",
            f"GRAPH_DATABASE_NAME={POSTGRES_DB}",
        ]

    if stack == "neo4j_postgres":
        if not neo4j_host or not neo4j_bolt_port:
            raise ValueError("Neo4j host/port are required for the neo4j stack")
        return _postgres_relational_env(pg_host, pg_port) + [
            "VECTOR_DB_PROVIDER=lancedb",
            "GRAPH_DATABASE_PROVIDER=neo4j",
            f"GRAPH_DATABASE_URL=bolt://{neo4j_host}:{neo4j_bolt_port}",
            "GRAPH_DATABASE_NAME=neo4j",
            "GRAPH_DATABASE_USERNAME=neo4j",
            f"GRAPH_DATABASE_PASSWORD={NEO4J_PASSWORD}",
        ]

    raise ValueError(f"Unknown backend stack: {stack!r}")


def build_api_container_cmd(
    *,
    image_name: str,
    container_name: str,
    host_port: int,
    mock_llm_url: str,
    extra_env: list[str] | None = None,
) -> list[str]:
    """Build the ``docker run`` command for the API container.

    ``extra_env`` carries the per-stack backend selection (see
    :func:`db_env_for_stack`); the base env wires the mock LLM/embedding
    endpoints and skips the startup connection probe (the mock does not answer
    the ``"test"`` probe).
    """
    llm_endpoint = mock_llm_endpoint_for_container(mock_llm_url)
    env = [
        "ENV=local",
        "COGNEE_SKIP_CONNECTION_TEST=true",
        # Disable session memory so search/recall exercise the permanent graph only
        # (avoids extra session-analysis LLM calls during retrieval).
        "CACHING=false",
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
    env.extend(extra_env or [])

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
