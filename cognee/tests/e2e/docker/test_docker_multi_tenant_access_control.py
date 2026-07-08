"""End-to-end Docker smoke test for multi-tenant access control (#3369).

Proves tenant isolation end-to-end with ENABLE_BACKEND_ACCESS_CONTROL=true
against a running Cognee Docker container.

Opt-in: requires COGNEE_DOCKER_E2E_IMAGE env var pointing to a built image.

Example::

    COGNEE_DOCKER_E2E_IMAGE=cognee-backend:latest \\
    COGNEE_DOCKER_E2E_ENV_FILE=/path/to/cognee/.env \\
    uv run pytest cognee/tests/e2e/docker/test_docker_multi_tenant_access_control.py -s

Depends on: #3358 (Docker E2E harness).
TODO: Once #3358 is merged, refactor container lifecycle helpers into a shared
      conftest.py or harness module to avoid duplication with
      test_docker_remember_recall.py.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest


PASSWORD = "securepassword123!"

_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)([A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*\s*[=:]\s*)\S+"
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/-]+=*")
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")


def _redact_secrets(text: str) -> str:
    """Mask API keys, tokens, and passwords so failure output is safe to print."""
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1****", text)
    text = _BEARER_TOKEN_PATTERN.sub(r"\1****", text)
    return _OPENAI_KEY_PATTERN.sub("sk-****", text)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "cognee").exists():
            return parent
    raise RuntimeError("Could not find repository root.")


def _required_image() -> str:
    image = os.environ.get("COGNEE_DOCKER_E2E_IMAGE")
    if not image:
        pytest.skip("Set COGNEE_DOCKER_E2E_IMAGE to run the Docker E2E test.")
    return image


def _required_env_file() -> Path:
    env_file = Path(os.environ.get("COGNEE_DOCKER_E2E_ENV_FILE", _repo_root() / ".env"))
    if not env_file.exists():
        pytest.skip(f"Set COGNEE_DOCKER_E2E_ENV_FILE; {env_file} does not exist.")
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
    result = _run(
        ["docker", "logs", "--tail", str(tail), container_name],
        check=False,
        timeout=30,
    )
    return _redact_secrets((result.stdout + "\n" + result.stderr).strip())


def _start_container(image: str, env_file: Path, container_name: str, port: int) -> str:
    network = os.environ.get("COGNEE_DOCKER_E2E_NETWORK", "bridge").strip().lower()
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
            "ENABLE_BACKEND_ACCESS_CONTROL=true",
            "-e",
            "REQUIRE_AUTHENTICATION=true",
        ]
    )

    command.extend(
        [
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            f"mkdir -p {tmp_root}/system/databases {tmp_root}/data && /app/entrypoint.sh",
        ]
    )

    _run(command, timeout=120)
    return f"http://127.0.0.1:{port}"


def _wait_until_ready(base_url: str, container_name: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    logs = _container_logs(container_name, tail=100)
    raise RuntimeError(f"Container {container_name} failed to become healthy.\nLogs:\n{logs}")


def _cleanup_container(container_name: str) -> None:
    _run(["docker", "rm", "-f", container_name], check=False, timeout=30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def running_multi_tenant_container():
    """Start a Cognee container with access control enabled; tear down after."""
    image = _required_image()
    env_file = _required_env_file()
    port = _free_port()
    container_name = f"cognee-e2e-multitenant-{uuid.uuid4().hex[:8]}"

    _cleanup_container(container_name)
    base_url = _start_container(image, env_file, container_name, port)
    try:
        _wait_until_ready(base_url, container_name, timeout_seconds=120)
        yield base_url
    finally:
        _cleanup_container(container_name)


def _register_and_login_http(client: httpx.Client, email: str, password: str) -> str:
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    assert reg_resp.status_code in (201, 400), f"Register failed: {reg_resp.text}"

    login_resp = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    return login_resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_docker_multi_tenant_access_control(running_multi_tenant_container):
    """Full isolation + sharing flow against a real running container."""
    base_url = running_multi_tenant_container
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        uid = uuid.uuid4().hex[:8]
        email_a = f"docker_a_{uid}@example.com"
        email_b = f"docker_b_{uid}@example.com"

        token_a = _register_and_login_http(client, email_a, PASSWORD)
        token_b = _register_and_login_http(client, email_b, PASSWORD)

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 1. User A creates dataset_a
        resp_a = client.post(
            "/api/v1/datasets",
            json={"name": "dataset_a"},
            headers=headers_a,
        )
        assert resp_a.status_code == 200
        ds_a_id = resp_a.json()["id"]

        # 2. User B creates dataset_b
        resp_b = client.post(
            "/api/v1/datasets",
            json={"name": "dataset_b"},
            headers=headers_b,
        )
        assert resp_b.status_code == 200
        ds_b_id = resp_b.json()["id"]

        # 3. Assert User B cannot see dataset_a
        list_b = client.get("/api/v1/datasets", headers=headers_b)
        assert list_b.status_code == 200
        ds_ids_b = [d["id"] for d in list_b.json()]
        assert ds_b_id in ds_ids_b
        assert ds_a_id not in ds_ids_b

        # 4. Assert User B search on dataset_a returns no data.
        #    Ideal (issue spec): 200 + [] for no-info-leak.
        #    Current behaviour: PermissionDeniedError → 403.
        #    TODO: For true no-info-leak, search should return 200 + [].
        search_b = client.post(
            "/api/v1/search",
            json={
                "query": "secret info",
                "searchType": "GRAPH_COMPLETION",
                "dataset_ids": [ds_a_id],
            },
            headers=headers_b,
        )
        if search_b.status_code == 200:
            assert search_b.json() == [], (
                f"Unauthorized search must return empty [], got: {search_b.json()}"
            )
        else:
            assert search_b.status_code == 403, (
                f"Expected 403 or 200 (empty), got {search_b.status_code}: {search_b.text}"
            )

        # 5. Per-user data isolation: User B cannot list dataset_a's data.
        data_b = client.get(f"/api/v1/datasets/{ds_a_id}/data", headers=headers_b)
        if data_b.status_code == 200:
            assert data_b.json() == [], f"User B must not see User A's data, got: {data_b.json()}"
        else:
            assert data_b.status_code in (401, 403, 404)

        # 6. User A shares dataset_a with User B
        get_user_b = client.post(
            "/api/v1/users/get-user-id",
            json={"email": email_b},
            headers=headers_a,
        )
        assert get_user_b.status_code == 200
        user_b_id = get_user_b.json()["user_id"]

        grant_resp = client.post(
            f"/api/v1/permissions/datasets/{user_b_id}?permission_name=read",
            json=[ds_a_id],
            headers=headers_a,
        )
        assert grant_resp.status_code == 200

        # 7. Assert User B can now see dataset_a
        list_b_after = client.get("/api/v1/datasets", headers=headers_b)
        assert list_b_after.status_code == 200
        assert ds_a_id in [d["id"] for d in list_b_after.json()]


def test_docker_unauthenticated_rejected(running_multi_tenant_container):
    """Unauthenticated requests are rejected when access control is on."""
    base_url = running_multi_tenant_container
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        resp = client.get("/api/v1/datasets")
        assert resp.status_code == 401, (
            f"Expected 401 for unauthenticated request, got {resp.status_code}"
        )
