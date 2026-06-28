import pytest
import subprocess
import time
import docker
import httpx
import pathlib
import sys
import pytest_asyncio
from cognee.tests.deployment.helpers.health import wait_for_health

REPO_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent)

@pytest.fixture(scope="session")
def mock_llm_server():
    """Start FastAPI mock LLM server on port 11434"""
    print("🚀 Starting mock LLM server...")

    log_file = open("mock_server.log", "w")
    process = subprocess.Popen(
        [sys.executable, "-m", "cognee.tests.deployment.mock_llm.server"],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    time.sleep(3)
    wait_for_health("http://127.0.0.1:11434/health", timeout=30)

    print("✓ Mock LLM server ready")
    try:
        yield process
    finally:
        print("🛑 Stopping mock LLM server...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            log_file.close()

@pytest.fixture(scope="function")
def running_container(mock_llm_server):
    """Start Cognee container with mock LLM pointed at localhost"""
    print("🚀 Starting Cognee container...")

    client = docker.from_env()

    container = client.containers.run(
        "cognee/cognee:main",
        detach=True,
        environment={
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "mock",
            "LLM_ENDPOINT": "http://host.docker.internal:11434",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_ENDPOINT": "http://host.docker.internal:11434",
            "MOCK_LLM_MODE": "replay",
        },
        ports={"8000/tcp": 8000},
        extra_hosts={"host.docker.internal": "host-gateway"},
    )

    wait_for_health("http://127.0.0.1:8000/health", timeout=300)

    print("✓ Cognee container ready")
    try:
        yield container
    finally:
        print("🛑 Stopping container...")
        container.stop()
        container.remove()

@pytest_asyncio.fixture(scope="function")
async def api_client(running_container):
    """httpx client + auth helper"""
    client = httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=300.0)

    await client.post("/api/v1/auth/register", json={"email": "test@example.com", "password": "test123"})
    response = await client.post("/api/v1/auth/login", data={"username": "test@example.com", "password": "test123"})
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    try:
        yield client
    finally:
        await client.aclose()

@pytest.fixture(scope="session")
def build_image():
    """Build Docker image locally"""
    subprocess.run(
        ["docker", "build", "-t", "cognee/cognee:local", "."],
        check=True,
    )
    return "cognee/cognee:local"

@pytest.fixture(scope="session")
def pull_image():
    """Pull pre-built Docker image"""
    subprocess.run(
        ["docker", "pull", "cognee/cognee:main"],
        check=True,
    )
    return "cognee/cognee:main"