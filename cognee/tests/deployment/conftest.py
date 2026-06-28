"""Pytest fixtures for deployment end-to-end tests.

Adapted from PR #3563's harness and extended for the backend-DB matrix (T10,
issue #3368): the ``running_container`` fixture is parametrized by a backend
"stack" key and brings up the matching database containers before the API
container.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid

import httpx
import pytest
import pytest_asyncio

from cognee.tests.deployment.golden_flow import run_golden_flow, run_remember_recall_flow
from cognee.tests.deployment.helpers import (
    build_api_container_cmd,
    db_env_for_stack,
    db_host_for_container,
    get_free_port,
    resolve_docker_image,
    start_neo4j_container,
    start_postgres_container,
    stop_container,
    stream_container_logs,
    wait_for_health,
)
from cognee.tests.deployment.mock_llm import start_mock_llm_server, stop_mock_llm_server

# Quiet the very chatty httpx/httpcore wire logs so test output stays readable.
for _noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

DEFAULT_STACK = "sqlite_lancedb_kuzu"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--image",
        action="store",
        default=None,
        help="Docker image for API deployment tests: build, pull, or an image name",
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


@pytest.fixture
def running_container(request, mock_llm_server, image_name):
    """Start the API container (plus any DB containers) for a backend stack.

    Parametrized indirectly with a stack key understood by
    :func:`cognee.tests.deployment.helpers.db_env_for_stack`. Defaults to the
    file-based SQLite stack when used without parametrization.
    """
    stack = getattr(request, "param", DEFAULT_STACK)

    if _use_real_llm() and mock_llm_server is None:
        pytest.skip("Real LLM mode requires DEPLOYMENT_USE_REAL_LLM secrets to be configured.")

    started: list[str] = []
    api_container_name: str | None = None
    failed = False
    pg_host = pg_port = neo4j_host = neo4j_bolt = None

    try:
        if stack in ("postgres_pgvector_postgresgraph", "neo4j_postgres"):
            pg_name, pg_port = start_postgres_container()
            started.append(pg_name)
            pg_host = db_host_for_container()
        if stack == "neo4j_postgres":
            neo4j_name, neo4j_bolt = start_neo4j_container()
            started.append(neo4j_name)
            neo4j_host = db_host_for_container()

        db_env = db_env_for_stack(
            stack,
            pg_host=pg_host,
            pg_port=pg_port,
            neo4j_host=neo4j_host,
            neo4j_bolt_port=neo4j_bolt,
        )

        host_port = get_free_port()
        api_container_name = f"cognee-test-{uuid.uuid4().hex[:8]}"
        cmd = build_api_container_cmd(
            image_name=image_name,
            container_name=api_container_name,
            host_port=host_port,
            mock_llm_url=mock_llm_server or "",
            extra_env=db_env,
        )
        started.append(api_container_name)

        print(f"Starting [{stack}] container {api_container_name} on port {host_port}...")
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed to start: {proc.stderr}")

        url = f"http://127.0.0.1:{host_port}"
        wait_for_health(f"{url}/health", timeout=180.0)
        yield {
            "url": url,
            "port": host_port,
            "container_name": api_container_name,
            "stack": stack,
        }
    except Exception:
        failed = True
        raise
    finally:
        test_failed = getattr(getattr(request.node, "rep_call", None), "failed", False)
        if (failed or test_failed) and api_container_name:
            print(f"\n--- CONTAINER LOGS FOR {api_container_name} ({stack}) ---")
            stream_container_logs(api_container_name)
        for name in reversed(started):
            stop_container(name)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest_asyncio.fixture
async def api_client(running_container):
    async with httpx.AsyncClient(base_url=running_container["url"], timeout=120.0) as client:
        yield client


@pytest.fixture
def golden_flow():
    return run_golden_flow


@pytest.fixture
def remember_recall_flow():
    return run_remember_recall_flow
