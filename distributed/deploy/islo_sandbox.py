"""
Cognee on Islo — deploy Cognee API inside an Islo cloud sandbox.

Islo (https://islo.dev) provides isolated cloud sandbox VMs for autonomous
agents, built by the Incredibuild team. This script creates a sandbox,
installs Cognee, starts the API server, and prints a temporary public
share URL for the API.

Prerequisites:
    pip install islo

    Set environment variables:
        ISLO_API_KEY  — your Islo API key (from https://app.islo.dev/api-keys)
        LLM_API_KEY   — your LLM provider API key

Usage:
    python distributed/deploy/islo_sandbox.py

Note:
    The sandbox name is fixed ("cognee-api"). Re-running the script while a
    previous deployment still exists fails with a name conflict — delete the
    old sandbox first:
        python -c "from islo import Islo; Islo().sandboxes.delete_sandbox('cognee-api')"
"""

import os
import sys
import time

from islo import Islo
from islo.errors import ConflictError
from islo.types import LifecyclePolicy

DEFAULT_IMAGE = "ghcr.io/islo-labs/islo-runner:latest"
SANDBOX_NAME = "cognee-api"
API_PORT = 8000
TERMINAL_EXEC_STATUSES = {"completed", "failed", "timeout"}

# The islo-runner image ships Python without pip/ensurepip and marks the system
# environment as PEP 668 "externally managed", so Cognee is installed into a
# dedicated virtualenv (bootstrapped via apt) rather than the system interpreter.
VENV_PATH = "/opt/cognee-venv"
VENV_PYTHON = f"{VENV_PATH}/bin/python"
_INSTALL_SCRIPT = (
    "set -e\n"
    "export DEBIAN_FRONTEND=noninteractive\n"
    "apt-get update -qq\n"
    "apt-get install -y -qq python3-venv\n"
    f"python3 -m venv {VENV_PATH}\n"
    f"{VENV_PATH}/bin/pip install --upgrade pip\n"
    f"{VENV_PATH}/bin/pip install 'cognee[api]'\n"
)

_HEALTHCHECK_SNIPPET = (
    "import urllib.request\n"
    "try:\n"
    "    urllib.request.urlopen('http://localhost:8000/health', timeout=5)\n"
    "    print('OK')\n"
    "except Exception:\n"
    "    print('WAITING')\n"
)


def run_command(
    client,
    sandbox_name: str,
    command: list,
    *,
    env: dict = None,
    workdir: str = None,
    timeout_secs: int = None,
    poll_interval: float = 2.0,
    max_wait: float = None,
    label: str = "",
):
    """Run a command in the sandbox and poll until it reaches a terminal status.

    Islo exec is asynchronous: ``exec_in_sandbox`` starts the command and returns
    an exec ID, then ``get_exec_result`` is polled until the status is one of
    ``completed``, ``failed``, or ``timeout``. Returns the final ExecResultResponse.

    ``timeout_secs`` is only a server-side hint, so polling is also bounded by a
    client-side deadline: ``max_wait`` seconds if given, otherwise derived from
    ``timeout_secs`` plus a margin (10 minutes when neither is set). Raises
    RuntimeError if the exec never reaches a terminal status in time.
    """
    started = client.sandboxes.exec_in_sandbox(
        sandbox_name,
        command=command,
        env=env,
        workdir=workdir,
        timeout_secs=timeout_secs,
    )

    if max_wait is None:
        max_wait = timeout_secs + 60.0 if timeout_secs else 600.0
    deadline = time.monotonic() + max_wait

    while True:
        result = client.sandboxes.get_exec_result(sandbox_name, started.exec_id)
        if result.status in TERMINAL_EXEC_STATUSES:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Command {command!r} did not reach a terminal status within {max_wait:.0f}s "
                f"(last status: '{result.status}')"
            )
        time.sleep(poll_interval)

    prefix = f"[{label}] " if label else ""
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"{prefix}{line}", flush=True)
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"{prefix}{line}", file=sys.stderr, flush=True)
    if result.truncated:
        print(f"{prefix}(output truncated)", flush=True)

    return result


def wait_for_sandbox_running(
    client,
    sandbox_name: str,
    timeout: float = 300,
    poll_interval: float = 3.0,
):
    """Poll the sandbox until it is running. Returns the SandboxResponse."""
    deadline = time.monotonic() + timeout
    while True:
        sandbox = client.sandboxes.get_sandbox(sandbox_name)
        if sandbox.status == "running":
            return sandbox
        if sandbox.status in {"failed", "stopped", "deleted"}:
            raise RuntimeError(f"Sandbox '{sandbox_name}' entered state '{sandbox.status}'")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Sandbox '{sandbox_name}' not running after {timeout}s")
        print(f"  Sandbox status: {sandbox.status}", flush=True)
        time.sleep(poll_interval)


def wait_for_server_health(
    client,
    sandbox_name: str,
    retries: int = 30,
    delay: float = 3.0,
) -> bool:
    """Poll the Cognee API /health endpoint inside the sandbox until it responds."""
    for attempt in range(retries):
        result = run_command(
            client,
            sandbox_name,
            ["python3", "-c", _HEALTHCHECK_SNIPPET],
            timeout_secs=30,
            poll_interval=1.0,
            label="health",
        )
        if "OK" in result.stdout:
            print("Server is ready!")
            return True
        print(f"  ({attempt + 1}) waiting for server...", flush=True)
        time.sleep(delay)

    # Print server log for debugging
    run_command(
        client,
        sandbox_name,
        ["bash", "-lc", "tail -n 30 /tmp/cognee-server.log"],
        timeout_secs=30,
        label="server-log",
    )
    print("WARNING: Server may not be ready yet.")
    return False


def deploy_cognee():
    """Create an Islo sandbox, start the Cognee API server, and print a share URL."""
    if not os.environ.get("ISLO_API_KEY"):
        raise ValueError("ISLO_API_KEY environment variable is required")
    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    # The client picks up ISLO_API_KEY from the environment automatically.
    client = Islo()

    print("Creating Islo sandbox for Cognee...")
    try:
        sandbox = client.sandboxes.create_sandbox(
            name=SANDBOX_NAME,
            image=DEFAULT_IMAGE,
            vcpus=2,
            memory_mb=4096,
            disk_gb=10,
            env={
                "LLM_API_KEY": llm_api_key,
                "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
                "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
                "HOST": "0.0.0.0",
            },
            lifecycle=LifecyclePolicy(pause_after_idle=3600, auto_resume="on_activity"),
        )
    except ConflictError as error:
        raise RuntimeError(
            f"A sandbox named '{SANDBOX_NAME}' already exists (likely from a previous run). "
            "Delete it first, then re-run this script:\n"
            f'  python -c "from islo import Islo; '
            f"Islo().sandboxes.delete_sandbox('{SANDBOX_NAME}')\""
        ) from error
    print(f"Sandbox created: {sandbox.id}\n")

    print("Waiting for sandbox to start...")
    wait_for_sandbox_running(client, SANDBOX_NAME)

    # Step 1: Install Cognee
    print("=== Installing Cognee ===")
    install = run_command(
        client,
        SANDBOX_NAME,
        ["bash", "-lc", _INSTALL_SCRIPT],
        timeout_secs=1800,
        label="install",
    )
    if install.status != "completed" or install.exit_code not in (0, None):
        raise RuntimeError(
            f"Failed to install cognee (status '{install.status}', exit {install.exit_code}): "
            f"{install.stderr}"
        )
    print()

    # Step 2: Start the server as a background daemon
    print("=== Starting Cognee API server ===")
    run_command(
        client,
        SANDBOX_NAME,
        [
            "bash",
            "-lc",
            f"nohup {VENV_PYTHON} -m uvicorn cognee.api.client:app --host 0.0.0.0 --port {API_PORT} "
            "> /tmp/cognee-server.log 2>&1 & echo started",
        ],
        timeout_secs=30,
        label="server",
    )

    print("Waiting for server to start...", flush=True)
    wait_for_server_health(client, SANDBOX_NAME)

    # Step 3: Create a temporary public share for the API port
    share = client.shares.create_share(SANDBOX_NAME, port=API_PORT, ttl_seconds=86400)

    print("\nCognee is running!")
    print(f"  Sandbox: {SANDBOX_NAME} ({sandbox.id})")
    print(f"\n  API URL: {share.url}")
    print(f"  Health:  {share.url}/health")
    print(f"  Docs:    {share.url}/docs")
    if share.expires_at:
        print(f"  (share URL expires at {share.expires_at})")
    else:
        print("  (share URL expires in 24 hours)")
    print("\nTo stop the sandbox:")
    print(f"  python -c \"from islo import Islo; Islo().sandboxes.stop_sandbox('{SANDBOX_NAME}')\"")
    print("To delete it:")
    print(
        f"  python -c \"from islo import Islo; Islo().sandboxes.delete_sandbox('{SANDBOX_NAME}')\""
    )

    return sandbox


if __name__ == "__main__":
    deploy_cognee()
