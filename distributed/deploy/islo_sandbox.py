"""
Cognee on Islo — deploy Cognee API with the Islo CLI.

Islo (https://islo.dev) provides isolated cloud sandbox VMs for autonomous
agents, built by the Incredibuild team. This script uses the official Islo CLI
to create a sandbox, install Cognee, start the API server, and print a temporary
public share URL.

Prerequisites:
    Install the Islo CLI:
        curl -fsSL https://islo.dev/install.sh | bash

    Authenticate once with ``islo login``. For CI, create a key with
    ``islo api-key create cognee-ci --expires 90`` and set ``ISLO_API_KEY``.

    Set your LLM provider API key:
        export LLM_API_KEY=your-key

Usage:
    python distributed/deploy/islo_sandbox.py

Note:
    The sandbox name is fixed ("cognee-api"). Re-running the script while a
    previous deployment still exists fails with a name conflict — delete the
    old sandbox first:
        islo rm cognee-api --force
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

DEFAULT_IMAGE = "ghcr.io/islo-labs/islo-runner:latest"
SANDBOX_NAME = "cognee-api"
API_PORT = 8000

# The islo-runner image ships Python without pip/ensurepip and marks the system
# environment as PEP 668 "externally managed", so Cognee is installed into a
# dedicated virtualenv (bootstrapped via apt) rather than the system interpreter.
VENV_PATH = "/opt/cognee-venv"
VENV_PYTHON = f"{VENV_PATH}/bin/python"
_DEPLOY_SCRIPT = (
    "set -e\n"
    "export DEBIAN_FRONTEND=noninteractive\n"
    "apt-get update -qq\n"
    "apt-get install -y -qq python3-venv\n"
    f"python3 -m venv {VENV_PATH}\n"
    f"{VENV_PATH}/bin/pip install --upgrade pip\n"
    f"{VENV_PATH}/bin/pip install 'cognee[api]'\n"
    f"nohup {VENV_PYTHON} -m uvicorn cognee.api.client:app "
    f"--host 0.0.0.0 --port {API_PORT} > /tmp/cognee-server.log 2>&1 &\n"
    "echo started\n"
)

_HEALTHCHECK_SNIPPET = (
    "import sys, urllib.request\n"
    "try:\n"
    f"    urllib.request.urlopen('http://localhost:{API_PORT}/health', timeout=5)\n"
    "except Exception:\n"
    "    sys.exit(1)\n"
    "print('OK')\n"
)


def run_islo(
    args: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
    echo: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an Islo CLI command and return its completed process."""
    command = ["islo", *args]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Islo command timed out after {timeout:.0f}s: {args!r}") from error

    if echo:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
                flush=True,
            )

    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Islo command failed (exit {result.returncode}){suffix}")

    return result


def _parse_json(result: subprocess.CompletedProcess[str], description: str) -> dict[str, Any]:
    """Parse a JSON object returned by the CLI with an actionable error."""
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(f"Islo returned invalid JSON while reading {description}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Islo returned an unexpected response while reading {description}")
    return value


def _write_env_file(values: dict[str, str]) -> str:
    """Write sandbox variables to a mode-0600 temporary dotenv file."""
    descriptor, path = tempfile.mkstemp(prefix="cognee-islo-", suffix=".env", text=True)
    try:
        with os.fdopen(descriptor, "w") as env_file:
            for key, value in values.items():
                env_file.write(f"{key}={json.dumps(value)}\n")
        os.chmod(path, 0o600)
    except Exception:
        os.unlink(path)
        raise
    return path


def _check_prerequisites() -> str:
    """Validate the local CLI, authentication, and required provider key."""
    if shutil.which("islo") is None:
        raise RuntimeError(
            "Islo CLI is required. Install it with: curl -fsSL https://islo.dev/install.sh | bash"
        )

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    status_result = run_islo(["status", "--output", "json"], check=False, echo=False)
    if status_result.returncode != 0:
        raise RuntimeError(
            "Islo authentication failed. Run 'islo login' or create a CI key with "
            "'islo api-key create cognee-ci --expires 90', then set ISLO_API_KEY."
        )
    status = _parse_json(status_result, "authentication status")
    if not status.get("auth", {}).get("authenticated"):
        raise RuntimeError(
            "Islo is not authenticated. Run 'islo login' or create a CI key with "
            "'islo api-key create cognee-ci --expires 90', then set ISLO_API_KEY."
        )

    return llm_api_key


def _sandbox_exists() -> bool:
    """Return whether the fixed deployment sandbox already exists."""
    result = run_islo(
        ["status", SANDBOX_NAME, "--output", "json"],
        check=False,
        echo=False,
    )
    return result.returncode == 0


def wait_for_server_health(retries: int = 30, delay: float = 3.0) -> bool:
    """Poll the Cognee API /health endpoint inside the sandbox."""
    for attempt in range(retries):
        result = run_islo(
            [
                "use",
                SANDBOX_NAME,
                "--no-config",
                "--output",
                "plain",
                "--",
                "python3",
                "-c",
                _HEALTHCHECK_SNIPPET,
            ],
            check=False,
            timeout=30,
            echo=False,
        )
        if result.returncode == 0 and result.stdout.strip().splitlines()[-1:] == ["OK"]:
            print("Server is ready!")
            return True
        print(f"  ({attempt + 1}) waiting for server...", flush=True)
        time.sleep(delay)

    run_islo(
        [
            "use",
            SANDBOX_NAME,
            "--no-config",
            "--output",
            "plain",
            "--",
            "bash",
            "-lc",
            "tail -n 30 /tmp/cognee-server.log",
        ],
        check=False,
        timeout=30,
    )
    print("ERROR: Server did not become healthy.", file=sys.stderr)
    return False


def deploy_cognee() -> dict[str, Any]:
    """Create an Islo sandbox, start the Cognee API, and print a share URL."""
    llm_api_key = _check_prerequisites()

    if _sandbox_exists():
        raise RuntimeError(
            f"A sandbox named '{SANDBOX_NAME}' already exists (likely from a previous run). "
            f"Delete it with 'islo rm {SANDBOX_NAME} --force', then re-run this script."
        )

    env_path = _write_env_file(
        {
            "LLM_API_KEY": llm_api_key,
            "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
            "HOST": "0.0.0.0",
        }
    )
    print("Creating Islo sandbox and installing Cognee...")
    try:
        run_islo(
            [
                "use",
                SANDBOX_NAME,
                "--no-config",
                "--image",
                DEFAULT_IMAGE,
                "--cpu",
                "2",
                "--memory",
                "4096",
                "--disk",
                "10",
                "--pause-after-idle",
                "3600",
                "--auto-resume",
                "on_activity",
                "--run-as-user",
                "root",
                "--env-file",
                env_path,
                "--output",
                "plain",
                "--",
                "bash",
                "-lc",
                _DEPLOY_SCRIPT,
            ],
            timeout=1860,
        )
    finally:
        os.unlink(env_path)

    print("Waiting for server to start...", flush=True)
    if not wait_for_server_health():
        raise RuntimeError(
            "Cognee API server did not become healthy; see the server log printed above"
        )

    share_result = run_islo(
        ["share", SANDBOX_NAME, str(API_PORT), "--ttl", "24h", "--output", "json"],
        echo=False,
    )
    share = _parse_json(share_result, "share URL")
    share_url = share.get("url")
    if not isinstance(share_url, str) or not share_url:
        raise RuntimeError("Islo did not return a share URL")

    sandbox_result = run_islo(
        ["status", SANDBOX_NAME, "--output", "json"],
        echo=False,
    )
    sandbox = _parse_json(sandbox_result, "sandbox status")

    print("\nCognee is running!")
    print(f"  Sandbox: {SANDBOX_NAME}")
    print(f"\n  API URL: {share_url}")
    print(f"  Health:  {share_url}/health")
    print(f"  Docs:    {share_url}/docs")
    if share.get("expires_at"):
        print(f"  (share URL expires at {share['expires_at']})")
    else:
        print("  (share URL expires in 24 hours)")
    print("\nTo stop the sandbox:")
    print(f"  islo stop {SANDBOX_NAME}")
    print("To delete it:")
    print(f"  islo rm {SANDBOX_NAME} --force")

    return sandbox


if __name__ == "__main__":
    deploy_cognee()
