"""Local deployment harness for the Fly.io end-to-end test in ``test_fly_deploy.py``."""

import os
import time
import asyncio
import subprocess
from uuid import uuid4
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import requests

# Repo root is four levels up: cognee/tests/deployment/fly_harness.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "distributed" / "deploy" / "fly-deploy.sh"


def wait_for_health(url: str, timeout: int = 600, interval: int = 10) -> bool:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True
        except requests.RequestException as error:
            last_error = error
        time.sleep(interval)
    raise TimeoutError(
        f"Service at {url} did not become healthy within {timeout}s (last error: {last_error})"
    )


def deploy_fly_app(
    app_name: str,
    *,
    region: str = "iad",
    volume_size: int = 1,
    llm_env: dict | None = None,
) -> str:
    """Deploy a throwaway app via fly-deploy.sh (run from repo root) and return its URL."""
    if not DEPLOY_SCRIPT.exists():
        raise FileNotFoundError(f"Deploy script not found: {DEPLOY_SCRIPT}")

    env = os.environ.copy()
    env["FLY_APP_NAME"] = app_name
    env["FLY_REGION"] = region
    env["FLY_VOLUME_SIZE"] = str(volume_size)
    for key, value in (llm_env or {}).items():
        if value is not None:
            env[key] = str(value)

    if not env.get("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY must be set; fly-deploy.sh requires it.")

    subprocess.run(["bash", str(DEPLOY_SCRIPT)], cwd=str(REPO_ROOT), env=env, check=True)

    return f"https://{app_name}.fly.dev"


def destroy_fly_app(app_name: str) -> None:
    """Destroy the Fly app and its volume. Never raises (cleanup must not mask failures)."""
    try:
        subprocess.run(
            ["fly", "apps", "destroy", app_name, "--yes"],
            check=False,
            timeout=300,
        )
    except Exception as error:  # noqa: BLE001 - cleanup must never raise
        print(f"Failed to destroy Fly app {app_name}: {error}")


@asynccontextmanager
async def authed_client(
    base_url: str,
    email: str | None = None,
    password: str = "cognee-ci-pass-123!",
):
    """Yield an httpx client authenticated (register + login) against the deployed app."""
    email = email or f"ci-{uuid4().hex[:8]}@example.com"
    client = httpx.AsyncClient(base_url=base_url, timeout=300.0)
    try:
        await client.post("/api/v1/auth/register", json={"email": email, "password": password})
        login = await client.post(
            "/api/v1/auth/login",
            data={"username": email, "password": password},
        )
        login.raise_for_status()
        client.headers["Authorization"] = f"Bearer {login.json()['access_token']}"
        yield client
    finally:
        await client.aclose()


async def golden_flow(
    api_client: httpx.AsyncClient,
    *,
    dataset_name: str | None = None,
    expected_entity: str = "Alice",
    poll_attempts: int = 60,
    poll_interval: int = 5,
) -> bool:
    """Run add -> cognify -> search against a live Cognee API; assert the seeded entity returns.

    Signature-compatible with the shared ``golden_flow`` in #3596 (swap in once T0 merges).
    """
    dataset_name = dataset_name or f"fly_ci_{uuid4().hex[:8]}"
    known_doc = f"{expected_entity} works at Cognee and manages the AI memory platform."

    # 1. Add the known document (synchronous ingest by default).
    files = {"data": ("doc.txt", known_doc.encode("utf-8"), "text/plain")}
    add_response = await api_client.post(
        "/api/v1/add", files=files, data={"datasetName": dataset_name}
    )
    add_response.raise_for_status()
    dataset_id = add_response.json()["dataset_id"]

    # 2. Cognify by dataset id (a real UUID from the add response).
    cognify_response = await api_client.post("/api/v1/cognify", json={"dataset_ids": [dataset_id]})
    cognify_response.raise_for_status()

    # 3. Poll status until the cognify pipeline completes.
    status = None
    for _ in range(poll_attempts):
        status_response = await api_client.get(
            "/api/v1/datasets/status", params={"dataset": dataset_id}
        )
        status_response.raise_for_status()
        status = str(status_response.json().get(str(dataset_id), ""))
        if status == "DATASET_PROCESSING_COMPLETED":
            break
        if status == "DATASET_PROCESSING_ERRORED":
            raise RuntimeError(f"Cognify errored for dataset {dataset_id}")
        await asyncio.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"Cognify did not complete within {poll_attempts * poll_interval}s "
            f"(last status: {status!r})"
        )

    # 4. Search the graph.
    search_response = await api_client.post(
        "/api/v1/search",
        json={
            "query": "Who works at Cognee?",
            "search_type": "GRAPH_COMPLETION",
            "dataset_ids": [dataset_id],
        },
    )
    search_response.raise_for_status()

    # 5. Assert the seeded entity is present in the results.
    payload = search_response.json()
    results = payload if isinstance(payload, list) else payload.get("results", payload)
    if expected_entity not in str(results):
        raise AssertionError(
            f"Expected entity {expected_entity!r} not found in search results: {results}"
        )
    return True
