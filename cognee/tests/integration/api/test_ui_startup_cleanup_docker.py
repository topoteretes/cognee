"""Opt-in Docker lifecycle smoke test (no LLM or frontend dependencies).

COGNEE_TEST_DOCKER_CLEANUP=1 pytest cognee/tests/integration/api/test_ui_startup_cleanup_docker.py
Pre-pull alpine:3.22, or set COGNEE_CLEANUP_TEST_IMAGE to an existing lightweight image.
The container runs sleep instead of the MCP application; Docker lifecycle is real.
"""

import importlib
import os
import subprocess
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("COGNEE_TEST_DOCKER_CLEANUP") != "1",
    reason="Opt-in Docker daemon smoke test",
)


def test_failed_startup_removes_owned_container_and_preserves_other_container(
    monkeypatch, tmp_path
):
    ui = importlib.import_module("cognee.api.v1.ui.ui")
    image = os.environ.get("COGNEE_CLEANUP_TEST_IMAGE", "alpine:3.22")
    run, popen = subprocess.run, subprocess.Popen
    run(["docker", "image", "inspect", image], check=True, capture_output=True, timeout=15)
    external = f"cognee-cleanup-external-{uuid.uuid4().hex}"
    owned = []
    started = []

    def spawn(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            name = command[command.index("--name") + 1]
            owned.append(name)
            command = ["docker", "run", "--rm", "--name", name, image, "sleep", "120"]
        return popen(command, **kwargs)

    def run_command(command, **kwargs):
        if command[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(command, 0)
        return run(command, **kwargs)

    def running(name):
        result = run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def on_spawn(pid_and_name):
        _, name = pid_and_name
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if running(name):
                started.append(name)
                return
            time.sleep(0.1)
        pytest.fail("Smoke container did not start")

    run(
        ["docker", "run", "-d", "--rm", "--name", external, image, "sleep", "120"],
        check=True,
        capture_output=True,
        timeout=15,
    )
    try:
        monkeypatch.setattr(ui.subprocess, "Popen", spawn)
        monkeypatch.setattr(ui.subprocess, "run", run_command)
        monkeypatch.setattr(ui, "_check_required_ports", lambda _: (True, []))
        monkeypatch.setattr(ui, "find_frontend_path", lambda: tmp_path)
        monkeypatch.setattr(ui, "check_node_npm", lambda: (False, "simulated missing Node.js"))
        assert ui.start_ui(on_spawn, start_mcp=True, open_browser=False) is None
        assert len(started) == 1 and started == owned
        for name in owned:
            result = run(["docker", "inspect", name], capture_output=True, timeout=5, check=False)
            assert result.returncode != 0, "Startup leaked its container"
        assert running(external), "Startup cleanup stopped an externally managed container"
    finally:
        # Only test-created, unique names, including the external-service sentinel.
        for name in [*owned, external]:
            run(["docker", "rm", "--force", name], capture_output=True, timeout=10, check=False)
