"""End-to-end Docker smoke test for remember -> graph recall.

This test is intentionally opt-in because it starts Docker, calls the real
LLM/embedding provider configured in an env file, and can take a few minutes.

Example:
    COGNEE_DOCKER_E2E_IMAGE=cognee-pr2776:smoke \
    COGNEE_DOCKER_E2E_ENV_FILE=/path/to/cognee/.env \
    COGNEE_DOCKER_E2E_NETWORK=bridge \
    uv run pytest cognee/tests/e2e/docker/test_docker_remember_recall.py -s
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest


PASSWORD = "securepassword123!"
VERIFY_PHRASE = "crimson-river-2776"
VERIFY_NOTE = "Vela-117"

_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)([A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*\s*[=:]\s*)\S+"
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/-]+=*")
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")


def _redact_secrets(text: str) -> str:
    """Mask API keys, tokens, and passwords so failure output is safe to print in CI logs."""
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1****", text)
    text = _BEARER_TOKEN_PATTERN.sub(r"\1****", text)
    return _OPENAI_KEY_PATTERN.sub("sk-****", text)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "cognee").exists():
            return parent
    raise RuntimeError("Could not find repository root.")


def _env_file_has_any_key(env_file: Path, keys: set[str]) -> bool:
    if not env_file.exists():
        return False

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key in keys and value:
            return True

    return False


def _required_image() -> str:
    image = os.environ.get("COGNEE_DOCKER_E2E_IMAGE")
    if not image:
        pytest.skip("Set COGNEE_DOCKER_E2E_IMAGE to run the Docker E2E test.")
    return image


def _required_env_file() -> Path:
    env_file = Path(os.environ.get("COGNEE_DOCKER_E2E_ENV_FILE", _repo_root() / ".env"))
    if not env_file.exists():
        pytest.skip(f"Set COGNEE_DOCKER_E2E_ENV_FILE; {env_file} does not exist.")

    key_names = {"OPENAI_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"}
    if not any(os.environ.get(key) for key in key_names) and not _env_file_has_any_key(
        env_file, key_names
    ):
        pytest.skip(
            "Docker E2E test needs OPENAI_API_KEY, LLM_API_KEY, or EMBEDDING_API_KEY "
            "in the environment or env file."
        )

    return env_file


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(
    command: list[str], *, timeout: int = 60, check: bool = True
) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise AssertionError(f"Command failed ({result.returncode}): {' '.join(command)}\n{output}")
    return result


def _container_logs(container_name: str, *, tail: int = 300) -> str:
    """Return the container's recent logs with secret-looking values redacted."""
    result = _run(
        ["docker", "logs", "--tail", str(tail), container_name],
        check=False,
        timeout=30,
    )
    return _redact_secrets((result.stdout + "\n" + result.stderr).strip())


def _container_status(container_name: str) -> str:
    result = _run(
        [
            "docker",
            "inspect",
            "--format",
            "status={{.State.Status}} exit={{.State.ExitCode}} error={{.State.Error}}",
            container_name,
        ],
        check=False,
        timeout=30,
    )
    return (result.stdout + "\n" + result.stderr).strip()


def _start_container(image: str, env_file: Path, container_name: str, port: int) -> str:
    network = os.environ.get("COGNEE_DOCKER_E2E_NETWORK", "bridge").strip().lower()
    if network not in {"host", "bridge"}:
        pytest.fail("COGNEE_DOCKER_E2E_NETWORK must be 'host' or 'bridge'.")

    tmp_root = f"/tmp/cognee_docker_e2e_{container_name}"
    command = ["docker", "run", "-d", "--name", container_name]

    if network == "host":
        command.extend(["--network", "host"])
    else:
        command.extend(["-p", f"127.0.0.1:{port}:{port}"])

    command.extend(
        [
            "--env-file",
            str(env_file),
            "-v",
            f"{env_file}:/app/.env:ro",
            "-e",
            f"HTTP_PORT={port}",
            "-e",
            f"SYSTEM_ROOT_DIRECTORY={tmp_root}/system",
            "-e",
            f"DATA_ROOT_DIRECTORY={tmp_root}/data",
            "-e",
            f"CACHE_ROOT_DIRECTORY={tmp_root}/cache",
            "-e",
            f"COGNEE_LOGS_DIR={tmp_root}/logs",
            "-e",
            "GRAPH_DATABASE_PROVIDER=kuzu",
        ]
    )

    tiktoken_cache = os.environ.get("COGNEE_DOCKER_E2E_TIKTOKEN_CACHE")
    if tiktoken_cache:
        cache_path = Path(tiktoken_cache)
        if not cache_path.exists():
            pytest.fail(f"COGNEE_DOCKER_E2E_TIKTOKEN_CACHE does not exist: {cache_path}")
        command.extend(
            [
                "-v",
                f"{cache_path}:/tmp/tiktoken-cache:ro",
                "-e",
                "TIKTOKEN_CACHE_DIR=/tmp/tiktoken-cache",
            ]
        )

    command.extend(
        [
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "mkdir -p "
            f"{tmp_root}/system/databases {tmp_root}/data {tmp_root}/cache {tmp_root}/logs "
            "&& /app/entrypoint.sh",
        ]
    )

    _run(command, timeout=120)
    return f"http://127.0.0.1:{port}"


def _wait_until_ready(base_url: str, container_name: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    last_status = ""

    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"{response.status_code}: {response.text[:500]}"
        except httpx.HTTPError as exc:
            last_error = repr(exc)

        last_status = _container_status(container_name)
        if last_status.startswith("status=exited") or last_status.startswith("status=dead"):
            pytest.fail(
                "Cognee Docker container exited before becoming healthy: "
                f"{last_error}\ncontainer status: {last_status}\n"
                f"--- docker logs ---\n{_container_logs(container_name)}\n--- end docker logs ---"
            )

        time.sleep(2)

    pytest.fail(
        "Cognee Docker container did not become healthy: "
        f"{last_error}\ncontainer status: {last_status}\n"
        f"--- docker logs ---\n{_container_logs(container_name)}\n--- end docker logs ---"
    )


def _assert_env_is_available_in_container(container_name: str) -> None:
    check_command = (
        'test -r /app/.env && ([ -n "$OPENAI_API_KEY" ] || '
        '[ -n "$LLM_API_KEY" ] || [ -n "$EMBEDDING_API_KEY" ])'
    )
    _run(["docker", "exec", container_name, "/bin/sh", "-c", check_command], timeout=30)


def _auth_headers(client: httpx.Client, suffix: str) -> dict[str, str]:
    email = f"docker-e2e-{suffix}@example.com"
    register_response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert register_response.status_code in (201, 400), register_response.text

    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": PASSWORD},
        timeout=30,
    )
    assert login_response.status_code == 200, login_response.text

    token = login_response.json()["access_token"]
    assert token
    return {"Authorization": f"Bearer {token}"}


def _item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "content", "answer"):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)

    raw = item.get("raw")
    if isinstance(raw, dict):
        for key in ("text", "content", "value", "completion", "answer"):
            value = raw.get(key)
            if isinstance(value, str):
                parts.append(value)

    return "\n".join(parts)


def _all_result_text(results: list[dict[str, Any]]) -> str:
    return "\n".join(_item_text(item) for item in results)


def test_docker_remember_then_graph_recall_returns_remembered_fact() -> None:
    image = _required_image()
    env_file = _required_env_file()

    suffix = uuid.uuid4().hex[:10]
    container_name = f"cognee-docker-e2e-{suffix}"
    port = int(os.environ.get("COGNEE_DOCKER_E2E_PORT") or _free_port())
    base_url = _start_container(image, env_file, container_name, port)
    keep_container = os.environ.get("COGNEE_DOCKER_E2E_KEEP_CONTAINER") == "1"

    try:
        _wait_until_ready(
            base_url,
            container_name,
            timeout_seconds=int(os.environ.get("COGNEE_DOCKER_E2E_STARTUP_TIMEOUT", "240")),
        )
        _assert_env_is_available_in_container(container_name)

        dataset_name = f"docker_e2e_graph_{suffix}"
        fact = (
            f"Docker E2E graph fact: the graph verification phrase is {VERIFY_PHRASE}. "
            f"The calibration note {VERIFY_NOTE} belongs to this Docker graph E2E test."
        )
        query = f"What is the graph verification phrase for calibration note {VERIFY_NOTE}?"

        timeout = httpx.Timeout(
            connect=15,
            read=float(os.environ.get("COGNEE_DOCKER_E2E_REMEMBER_TIMEOUT", "420")),
            write=30,
            pool=15,
        )

        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            headers = _auth_headers(client, suffix)

            remember_response = client.post(
                "/api/v1/remember",
                headers=headers,
                data={
                    "datasetName": dataset_name,
                    "run_in_background": "false",
                    "chunks_per_batch": "10",
                },
                files={"data": ("docker-e2e-memory.txt", fact.encode("utf-8"), "text/plain")},
            )
            assert remember_response.status_code == 200, remember_response.text

            remember_result = remember_response.json()
            assert remember_result["status"] == "completed"
            assert remember_result["dataset_name"] == dataset_name
            assert remember_result["dataset_id"]
            assert remember_result["pipeline_run_id"]

            chunks_response = client.post(
                "/api/v1/recall",
                headers=headers,
                json={
                    "query": query,
                    "search_type": "CHUNKS",
                    "datasets": [dataset_name],
                    "top_k": 5,
                    "scope": "graph",
                },
            )
            assert chunks_response.status_code == 200, chunks_response.text

            chunk_results = chunks_response.json()
            assert chunk_results
            assert any(result.get("source") == "graph" for result in chunk_results)

            chunk_text = _all_result_text(chunk_results).lower()
            assert VERIFY_PHRASE in chunk_text
            assert VERIFY_NOTE.lower() in chunk_text

            graph_context_response = client.post(
                "/api/v1/recall",
                headers=headers,
                json={
                    "query": query,
                    "search_type": "GRAPH_COMPLETION",
                    "datasets": [dataset_name],
                    "top_k": 5,
                    "scope": "graph",
                    "only_context": True,
                },
            )
            assert graph_context_response.status_code == 200, graph_context_response.text

            graph_context_results = graph_context_response.json()
            assert graph_context_results
            assert any(result.get("source") == "graph" for result in graph_context_results)

            graph_context_text = _all_result_text(graph_context_results).lower()
            assert VERIFY_PHRASE in graph_context_text
            assert VERIFY_NOTE.lower() in graph_context_text
    except Exception:
        print("\n--- docker logs ---")
        print(_container_logs(container_name))
        print("--- end docker logs ---\n")
        raise
    finally:
        if not keep_container:
            _run(["docker", "stop", container_name], timeout=60, check=False)
            _run(["docker", "rm", container_name], timeout=60, check=False)
