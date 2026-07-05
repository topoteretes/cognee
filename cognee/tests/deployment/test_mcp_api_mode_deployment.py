"""Deployment test: MCP container in API mode, two-container topology.

Verifies the production topology:
    Backend container -> Shared Docker network -> MCP container -> API_URL

Uses the existing docker-compose pattern from the project:
    docker compose -f cognee/tests/deployment/docker-compose.test.yml up

Session-based remember is used so no LLM_API_KEY is required.

Usage:
    pytest cognee/tests/deployment/test_mcp_api_mode_deployment.py -v
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)

BACKEND_IMAGE = os.environ.get("COGNEE_TEST_BACKEND_IMAGE", "cognee-test-backend:latest")
MCP_IMAGE = os.environ.get("COGNEE_TEST_MCP_IMAGE", "cognee-test-mcp:latest")
COMPOSE_FILE = str(Path(__file__).resolve().parent / "docker-compose.test.yml")
_POLL_TIMEOUT = 90


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _docker_preflight() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker binary not found on PATH")
    try:
        subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.TimeoutExpired, OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"Docker daemon not reachable: {exc}")


def _run(args: list[str], *, capture=True, timeout=300, check=True):
    cmd = ["docker"] + args
    logger.debug("$ docker %s", " ".join(str(a) for a in cmd))
    result = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"docker {' '.join(args)} failed (rc={result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


def _image_exists(tag: str) -> bool:
    return bool(_run(["images", "-q", tag], check=False).stdout.strip())


def _container_logs(container_name: str, tail: int = 80) -> str:
    try:
        r = _run(["logs", "--tail", str(tail), container_name], check=False)
        return r.stdout or r.stderr
    except RuntimeError:
        return "(logs unavailable)"


def _resolve_host_port(container_name: str, container_port: str) -> int:
    r = _run(["port", container_name, container_port])
    m = re.search(r"0\.0\.0\.0:(\d+)", r.stdout)
    if not m:
        # Docker Desktop sometimes lists as 127.0.0.1
        m = re.search(r"127\.0\.0\.1:(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"Cannot resolve host port for {container_name}: {r.stdout}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# MCP JSON-RPC helpers (no MCP client library dependency)
# ---------------------------------------------------------------------------

def _mcp_call(mcp_url: str, method: str, params: dict | None = None) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params:
        payload["params"] = params
    resp = httpx.post(f"{mcp_url.rstrip('/')}/mcp", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _mcp_result(envelope: dict):
    if "error" in envelope:
        err = envelope["error"]
        raise RuntimeError(f"MCP error ({err.get('code')}): {err.get('message')}")
    return envelope.get("result")


def _mcp_tool_call(mcp_url: str, name: str, arguments: dict) -> dict:
    env = _mcp_call(mcp_url, "tools/call", {"name": name, "arguments": arguments})
    return _mcp_result(env)


def _mcp_list_tools(mcp_url: str) -> list[dict]:
    env = _mcp_call(mcp_url, "tools/list")
    result = _mcp_result(env)
    return result.get("tools", [])


def _mcp_tool_text(result: dict) -> str:
    parts = []
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# pytest fixture (module scope)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def deployment() -> dict:
    """Module-scoped fixture: builds images, starts compose stack, cleans up.

    Returns ``{"backend_url": ..., "mcp_url": ...}``.
    """
    _docker_preflight()
    repo_root = Path(__file__).resolve().parents[2]

    # Build images
    for tag, df in [(BACKEND_IMAGE, "Dockerfile"),
                    (MCP_IMAGE, "cognee-mcp/Dockerfile")]:
        if not _image_exists(tag):
            logger.info("Building %s ...", tag)
            _run(["build", "-t", tag, "-f", str(repo_root / df), str(repo_root)])

    # Start compose stack
    _run(["compose", "-f", COMPOSE_FILE, "up", "-d", "--wait", "--wait-timeout", str(_POLL_TIMEOUT)])

    urls = {}
    try:
        urls["backend_port"] = _resolve_host_port("cognee-test-backend", "8000")
        urls["mcp_port"] = _resolve_host_port("cognee-test-mcp", "8000")
        urls["backend_url"] = f"http://127.0.0.1:{urls['backend_port']}"
        urls["mcp_url"] = f"http://127.0.0.1:{urls['mcp_port']}"

        # Wait for backend health even though compose should have done it
        _await_backend(urls["backend_url"])

        yield urls
    except Exception:
        for name in ("cognee-test-backend", "cognee-test-mcp"):
            logs = _container_logs(name)
            if logs:
                logger.error("Container logs for %s:\n%s", name, logs)
        raise
    finally:
        _run(["compose", "-f", COMPOSE_FILE, "down", "-t", "5"], check=False, timeout=60)


def _await_backend(backend_url: str) -> None:
    deadline = time.monotonic() + 60
    last = ""
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{backend_url}/health", timeout=5)
            if r.is_success and r.json().get("status") == "ready":
                return
            last = f"HTTP {r.status_code}"
        except (httpx.RequestError, ValueError) as e:
            last = str(e)
        time.sleep(2)
    raise TimeoutError(f"Backend not ready within 60s. Last: {last}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMCPApiModeDeployment:

    def test_backend_health(self, deployment: dict) -> None:
        """1. Backend starts and /health returns ready."""
        r = httpx.get(f"{deployment['backend_url']}/health", timeout=10)
        assert r.status_code == 200
        assert r.json().get("status") == "ready"

    def test_mcp_connects_to_backend(self, deployment: dict) -> None:
        """2 + 3. MCP responds to JSON-RPC and remembers via the backend."""
        env = _mcp_call(deployment["mcp_url"], "ping")
        assert "result" in env, f"ping failed: {env}"

        result = _mcp_tool_call(deployment["mcp_url"], "remember", {
            "data": "Deployment connectivity check.",
            "session_id": "deploy-conn-check",
        })
        text = _mcp_tool_text(result)
        assert "Error" not in text, f"remember failed: {text}"
        assert "Stored in session cache" in text, f"Unexpected: {text}"

    def test_remember_recall_forget_roundtrip(self, deployment: dict) -> None:
        """4, 5, 6. Full remember -> recall -> forget -> recall round-trip."""
        session_id = f"deploy-rt-{uuid.uuid4().hex[:12]}"
        phrase = f"Deployment round-trip payload {session_id}"

        # remember
        result = _mcp_tool_call(deployment["mcp_url"], "remember", {
            "data": phrase,
            "session_id": session_id,
        })
        text = _mcp_tool_text(result)
        assert "Error" not in text, f"remember failed: {text}"
        assert "Stored in session cache" in text

        # recall should find it
        result = _mcp_tool_call(deployment["mcp_url"], "recall", {
            "query": phrase,
            "session_id": session_id,
            "top_k": 3,
        })
        text = _mcp_tool_text(result)
        assert "Error" not in text, f"recall failed: {text}"
        assert phrase in text, f"Phrase not found in recall:\n{text}"

        # forget
        result = _mcp_tool_call(deployment["mcp_url"], "forget", {
            "dataset": "main_dataset",
        })
        text = _mcp_tool_text(result)
        assert "Error" not in text, f"forget failed: {text}"

        # recall after forget — phrase should be gone
        result = _mcp_tool_call(deployment["mcp_url"], "recall", {
            "query": phrase,
            "session_id": session_id,
            "top_k": 3,
        })
        text = _mcp_tool_text(result)
        assert phrase not in text, (
            f"Phrase still present after forget:\n{text}"
        )

    def test_only_memory_tools_exposed(self, deployment: dict) -> None:
        """7. Only remember, recall, forget exposed in API mode."""
        tools = _mcp_list_tools(deployment["mcp_url"])
        names = {t["name"] for t in tools}

        for required in ("remember", "recall", "forget"):
            assert required in names, f"Required tool missing: {required}"

        forbidden = {
            "cognify", "search", "delete", "prune", "improve",
            "list_data", "get_document", "get_chunk_neighbors",
            "delete_dataset", "cognify_status", "save_interaction",
            "visualize_graph_ui", "upload_file_ui", "open_cognee_workspace",
            "cognify_file", "list_datasets_json", "list_dataset_data_json",
            "get_client_info_json", "create_dataset_json",
        }
        present = names & forbidden
        assert not present, f"Forbidden tools exposed: {sorted(present)}"
