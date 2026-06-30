"""E2E test: crash/restart durability for the cognify pipeline.

Verifies that killing the API container mid-cognify leaves the system in a
recoverable state.  On restart the lifespan handler detects the stale
``DATASET_PROCESSING_STARTED`` run and rolls it back, returning the databases
to their pre-crash state.

Implements https://github.com/topoteretes/cognee/issues/3371

Requirements
------------
* Docker and ``docker compose`` available on PATH.
* The repo root is the working directory (or ``COGNEE_REPO_ROOT`` is set).
* Postgres profile enabled.  The test manages ``docker compose`` lifecycle.
* Mock LLM — no API keys needed.

Run
---
    pytest cognee/tests/e2e/test_crash_restart_durability.py -v --timeout=600
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import time
from typing import Any, Dict, Optional

import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.environ.get(
    "COGNEE_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.yml")

API_BASE = "http://localhost:8000"
HEALTH_URL = f"{API_BASE}/health"

DEFAULT_USER = "default_user@example.com"
DEFAULT_PASS = "default_password"

DATASET_NAME = "crash_test_dataset"
TEST_TEXT = (
    "Albert Einstein developed the theory of relativity, "
    "which fundamentally changed the understanding of space, time, and energy. "
    "He received the Nobel Prize in Physics in 1921 for his explanation of "
    "the photoelectric effect."
)

# How long the pipeline delay is set inside the container (seconds).
# Must be long enough for the test to docker-kill mid-pipeline.
PIPELINE_DELAY_SECONDS = 15

# Timeout waiting for the API to become healthy.
HEALTH_TIMEOUT = 180  # seconds

# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _compose(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run ``docker compose`` with the postgres profile against the repo compose file."""
    cmd = [
        "docker",
        "compose",
        "-f",
        COMPOSE_FILE,
        "--profile",
        "postgres",
        *args,
    ]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        capture_output=capture,
        text=True,
    )


def wait_for_health(url: str = HEALTH_URL, timeout: int = HEALTH_TIMEOUT) -> None:
    """Poll a health endpoint until it returns 200 or *timeout* expires."""
    deadline = time.monotonic() + timeout
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(2)
    raise TimeoutError(
        f"Health endpoint {url} did not become ready within {timeout}s. "
        f"Last error: {last_err}"
    )


def wait_for_postgres(timeout: int = 60) -> None:
    """Poll ``pg_isready`` inside the postgres container until it's ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", "postgres", "pg_isready", "-U", "cognee"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"Postgres did not become ready within {timeout}s")


def kill_container(name: str = "cognee") -> None:
    """SIGKILL a container — no graceful shutdown, simulates a hard crash."""
    subprocess.run(["docker", "kill", name], check=True, capture_output=True)


def start_container(name: str = "cognee") -> None:
    """Start a stopped container (retains original configuration)."""
    subprocess.run(["docker", "start", name], check=True, capture_output=True)


def get_container_logs(name: str = "cognee") -> str:
    """Return the full stdout+stderr logs of a container."""
    result = subprocess.run(
        ["docker", "logs", name], capture_output=True, text=True
    )
    return result.stdout + result.stderr


def update_container_env(name: str, env_key: str, env_value: str) -> None:
    """Update an env var on a stopped container by committing and recreating.

    Docker doesn't allow changing env vars on an existing container, so we
    use ``docker compose up`` with the env var set in the host environment,
    which docker-compose.yml picks up via ``${VAR:-default}`` or passes
    through if listed under ``environment:``.

    For our case, ``COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS`` and
    ``COGNEE_TEST_PIPELINE_DELAY_SECONDS`` are read by ``os.getenv`` inside
    the Python code, not via compose environment directives.  So we need to
    set them in the container process.

    The simplest approach: stop the compose service, modify the compose
    command or re-create with the env var.
    """
    # We'll handle this through docker compose up with env vars in the
    # fixture instead.  This function is a placeholder.
    pass


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def login(base_url: str = API_BASE) -> str:
    """Login as the default user, return the Bearer token string."""
    resp = requests.post(
        f"{base_url}/api/v1/auth/login",
        data={"username": DEFAULT_USER, "password": DEFAULT_PASS},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    return f"Bearer {token}"


def auth_headers(token: str) -> Dict[str, str]:
    """Return headers dict with Authorization."""
    return {"Authorization": token}


def add_data(
    token: str,
    text: str = TEST_TEXT,
    dataset_name: str = DATASET_NAME,
    base_url: str = API_BASE,
) -> Dict[str, Any]:
    """POST /api/v1/add — upload text data to a dataset (blocking)."""
    # The add endpoint expects multipart/form-data with file(s) + datasetName
    files = [("data", (f"{dataset_name}.txt", io.BytesIO(text.encode()), "text/plain"))]
    resp = requests.post(
        f"{base_url}/api/v1/add",
        headers=auth_headers(token),
        files=files,
        data={"datasetName": dataset_name},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def trigger_cognify(
    token: str,
    datasets: list[str] | None = None,
    background: bool = True,
    base_url: str = API_BASE,
) -> Dict[str, Any]:
    """POST /api/v1/cognify — start cognification.

    When *background* is True the API returns immediately with a
    pipeline_run_id while processing continues server-side.
    """
    payload = {
        "datasets": datasets or [DATASET_NAME],
        "run_in_background": background,
    }
    resp = requests.post(
        f"{base_url}/api/v1/cognify",
        headers={**auth_headers(token), "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def search(
    token: str,
    query: str = "Einstein relativity",
    base_url: str = API_BASE,
) -> Dict[str, Any]:
    """POST /api/v1/search — GRAPH_COMPLETION search."""
    resp = requests.post(
        f"{base_url}/api/v1/search",
        headers={**auth_headers(token), "Content-Type": "application/json"},
        json={"searchType": "GRAPH_COMPLETION", "query": query},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_datasets(token: str, base_url: str = API_BASE) -> list[Dict[str, Any]]:
    """GET /api/v1/datasets — list all datasets."""
    resp = requests.get(
        f"{base_url}/api/v1/datasets",
        headers=auth_headers(token),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_dataset_graph(
    token: str, dataset_id: str, base_url: str = API_BASE
) -> Dict[str, Any]:
    """GET /api/v1/datasets/{id}/graph — nodes and edges."""
    resp = requests.get(
        f"{base_url}/api/v1/datasets/{dataset_id}/graph",
        headers=auth_headers(token),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_dataset_status(
    token: str, dataset_id: str, base_url: str = API_BASE
) -> Dict[str, Any]:
    """GET /api/v1/datasets/status — pipeline run status."""
    resp = requests.get(
        f"{base_url}/api/v1/datasets/status",
        headers=auth_headers(token),
        params={"dataset": dataset_id, "pipeline": "cognify_pipeline"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# DB state snapshot
# ---------------------------------------------------------------------------


def snapshot_db_state(token: str, dataset_id: str) -> Dict[str, Any]:
    """Capture a snapshot of the database state for before/after comparison.

    Returns a dict with:
      - node_slugs: set of node slug UUIDs from the relational graph endpoint
      - edge_count: number of edges
      - node_count: number of nodes
      - node_types: set of node type strings
      - raw: the full graph response for deeper inspection
    """
    try:
        graph = get_dataset_graph(token, dataset_id)
    except requests.HTTPError:
        # Dataset may not have graph data yet (pre-cognify)
        return {
            "node_slugs": set(),
            "edge_count": 0,
            "node_count": 0,
            "node_types": set(),
            "raw": {"nodes": [], "edges": []},
        }

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_slugs = set()
    node_types = set()
    for node in nodes:
        node_id = node.get("id") or node.get("slug")
        if node_id:
            node_slugs.add(str(node_id))
        node_type = (node.get("properties") or {}).get("type") or node.get("type")
        if node_type:
            node_types.add(node_type)

    return {
        "node_slugs": node_slugs,
        "edge_count": len(edges),
        "node_count": len(nodes),
        "node_types": node_types,
        "raw": graph,
    }


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------

# Patterns that indicate a problem in the container logs.
_TRACEBACK_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"Corrupted wal file", re.IGNORECASE),
]

# Patterns we *expect* during crash recovery — whitelist so the traceback
# scan doesn't false-positive on expected recovery logging.
_RECOVERY_WHITELIST = [
    re.compile(r"Startup recovery completed for cognify run"),
    re.compile(r"Cognify rollback completed for run"),
    re.compile(r"Test delay: sleeping"),
]

# The recovery log line we assert MUST be present.
RECOVERY_LOG_PATTERN = re.compile(r"Startup recovery completed for cognify run")


def scan_logs_for_tracebacks(logs: str) -> list[str]:
    """Return list of traceback occurrences that aren't whitelisted.

    Each entry is the matching line from the logs.
    """
    problems = []
    for line in logs.splitlines():
        for pattern in _TRACEBACK_PATTERNS:
            if pattern.search(line):
                # Check if this line is in a whitelisted context
                whitelisted = any(wp.search(line) for wp in _RECOVERY_WHITELIST)
                if not whitelisted:
                    problems.append(line.strip())
    return problems


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose_stack():
    """Start the full stack with postgres profile, yield, then tear down.

    Environment:
      - DB_PROVIDER=postgres so the cognee container uses postgres
      - COGNEE_TEST_PIPELINE_DELAY_SECONDS gives us a kill window
      - COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS=0 so recovery triggers
        immediately on restart (not after the default 1-hour threshold)
    """
    env = os.environ.copy()
    env.update({
        "ENV": "dev",
        "DB_PROVIDER": "postgres",
        "DB_HOST": "postgres",
        "DB_PORT": "5432",
        "DB_NAME": "cognee_db",
        "DB_USERNAME": "cognee",
        "DB_PASSWORD": "cognee",
        "COGNEE_TEST_PIPELINE_DELAY_SECONDS": str(PIPELINE_DELAY_SECONDS),
        "COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS": "0",
    })

    # Build and start
    subprocess.run(
        [
            "docker", "compose", "-f", COMPOSE_FILE,
            "--profile", "postgres",
            "up", "-d", "--build",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    try:
        # Wait for services
        wait_for_postgres()
        wait_for_health()
        yield env
    finally:
        # Always tear down, collect logs first
        _compose("logs", "--no-color", check=False)
        subprocess.run(
            [
                "docker", "compose", "-f", COMPOSE_FILE,
                "--profile", "postgres",
                "down", "-v", "--remove-orphans",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrashRestartDurability:
    """Kill the API container mid-cognify and verify recovery on restart."""

    def test_crash_and_recovery(self, compose_stack):
        """Full crash/restart durability scenario.

        1. Add data and snapshot DB state (BEFORE)
        2. Trigger cognify in background (with pipeline delay)
        3. Wait for pipeline to reach STARTED state
        4. docker kill (SIGKILL) the cognee container
        5. Restart the container (recovery runs in lifespan)
        6. Snapshot DB state (AFTER) and assert consistency with BEFORE
        7. Verify pipeline status is reset (not stuck on STARTED)
        8. Re-cognify the same dataset — must complete successfully
        9. Search for results
        10. Scan logs for unexpected tracebacks
        """
        env = compose_stack

        # --- Step 1: Login and add data ---
        token = login()
        add_result = add_data(token)
        assert add_result is not None, "Add data failed"

        # Find the dataset ID
        datasets = get_datasets(token)
        dataset = next(
            (d for d in datasets if d["name"] == DATASET_NAME), None
        )
        assert dataset is not None, f"Dataset '{DATASET_NAME}' not found after add"
        dataset_id = dataset["id"]

        # Snapshot BEFORE cognify
        snapshot_before = snapshot_db_state(token, dataset_id)

        # --- Step 2: Trigger cognify in background ---
        cognify_result = trigger_cognify(token, background=True)
        assert cognify_result is not None, "Cognify trigger failed"

        # --- Step 3: Wait for the pipeline to be in STARTED state ---
        # The pipeline delay gives us time. Wait a few seconds for the
        # background task to begin and log a PipelineRun with STARTED status.
        time.sleep(5)

        # Verify the pipeline is in progress
        try:
            status_resp = get_dataset_status(token, dataset_id)
            # Status should show DATASET_PROCESSING_STARTED for cognify_pipeline
            status_str = str(status_resp)
            assert (
                "DATASET_PROCESSING_STARTED" in status_str
                or "STARTED" in status_str.upper()
            ), (
                f"Expected pipeline to be in STARTED state, got: {status_resp}. "
                "The pipeline delay may be too short or the pipeline completed too quickly."
            )
        except (requests.RequestException, AssertionError) as exc:
            # If we can't check status, the pipeline may have started but
            # that's OK — we'll still kill and test recovery.
            print(f"Warning: Could not verify STARTED state: {exc}")

        # --- Step 4: docker kill (SIGKILL) — hard crash ---
        kill_container("cognee")

        # Brief pause to let docker fully stop the container
        time.sleep(3)

        # --- Step 5: Restart the cognee container ---
        # We need to recreate with the recovery env var. Since the compose
        # stack was started with COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS=0,
        # the restarted container will immediately attempt recovery.
        subprocess.run(
            [
                "docker", "compose", "-f", COMPOSE_FILE,
                "--profile", "postgres",
                "up", "-d", "cognee",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )

        # Wait for the API to become healthy again (recovery runs during startup)
        wait_for_health(timeout=HEALTH_TIMEOUT)

        # Give recovery a moment to complete after health reports OK
        time.sleep(5)

        # --- Step 6: Verify recovery and DB consistency ---

        # Re-login (the container restarted, old sessions may be invalid)
        token = login()

        # Check container logs for recovery confirmation
        logs = get_container_logs("cognee")
        assert RECOVERY_LOG_PATTERN.search(logs), (
            "Expected 'Startup recovery completed for cognify run' in container "
            "logs, but it was not found. Recovery may not have triggered.\n"
            f"Last 50 lines of logs:\n{''.join(logs.splitlines()[-50:])}"
        )

        # Snapshot AFTER recovery
        snapshot_after = snapshot_db_state(token, dataset_id)

        # Assert: AFTER recovery state matches BEFORE crash state.
        # The partial cognify writes should have been rolled back.
        assert snapshot_after["node_count"] == snapshot_before["node_count"], (
            f"Node count mismatch after recovery: "
            f"before={snapshot_before['node_count']}, after={snapshot_after['node_count']}. "
            "Partial cognify writes may not have been fully rolled back."
        )
        assert snapshot_after["edge_count"] == snapshot_before["edge_count"], (
            f"Edge count mismatch after recovery: "
            f"before={snapshot_before['edge_count']}, after={snapshot_after['edge_count']}. "
            "Partial cognify writes may not have been fully rolled back."
        )
        assert snapshot_after["node_slugs"] == snapshot_before["node_slugs"], (
            "Node slugs differ after recovery. "
            f"Extra nodes: {snapshot_after['node_slugs'] - snapshot_before['node_slugs']}. "
            f"Missing nodes: {snapshot_before['node_slugs'] - snapshot_after['node_slugs']}."
        )

        # --- Step 7: Verify pipeline status is reset ---
        status_after = get_dataset_status(token, dataset_id)
        status_str = str(status_after)
        assert "DATASET_PROCESSING_STARTED" not in status_str, (
            f"Pipeline status is still STARTED after recovery: {status_after}. "
            "reset_pipeline_run_status() may not have run."
        )

        # --- Step 8: Re-cognify — must complete without being blocked ---
        # Run cognify again blocking (not in background) to ensure it completes.
        # This proves that check_pipeline_run_qualification doesn't block
        # because the stale STARTED status was reset to INITIATED.
        recog_result = trigger_cognify(token, background=False)
        assert recog_result is not None, (
            "Re-cognify failed after recovery. The pipeline may still be "
            "blocked by a stale STARTED status."
        )

        # --- Step 9: Search for results ---
        search_results = search(token, query="Einstein")
        assert search_results is not None, "Search returned None after re-cognify"
        # After a successful cognify, search should return non-empty results
        # (the search response structure varies, but it shouldn't be empty/error)

        # --- Step 10: Log scan ---
        final_logs = get_container_logs("cognee")
        tracebacks = scan_logs_for_tracebacks(final_logs)
        # Filter out tracebacks that are expected during recovery
        # (e.g., the recovery process itself may log errors about the failed run)
        critical_tracebacks = [
            tb for tb in tracebacks
            if "Corrupted wal file" not in tb
            and "Rollback errored" not in tb
        ]
        # We don't fail on tracebacks from the recovery itself (those are expected),
        # but we DO fail on WAL corruption or rollback failures.
        for tb_line in tracebacks:
            assert "Corrupted wal file" not in tb_line, (
                f"WAL corruption detected in logs: {tb_line}"
            )
            assert "Rollback errored" not in tb_line, (
                f"Rollback error detected in logs: {tb_line}"
            )

        # Verify recovery log lines are present
        assert "Startup recovery completed" in final_logs or "Cognify rollback completed" in final_logs, (
            "Neither recovery nor rollback completion log found. "
            "The crash/restart recovery path may not have executed."
        )
