"""
Cognee on boxd: deploy the Cognee API on a boxd machine.

boxd (https://boxd.sh?ref=cognee) runs microVMs: full Ubuntu machines with a
persistent disk and a public HTTPS URL, which hibernate when idle and wake on
the next request. This script uses the official boxd Python SDK to create a
machine, install Cognee, run the API server as a systemd service, and print
the machine's HTTPS URL.

Prerequisites:
    pip install boxd

    Create a boxd API key in the console or with the boxd CLI:
        boxd auth keys create cognee-deploy

    Set environment variables:
        BOXD_API_KEY  - your boxd API key
        LLM_API_KEY   - your LLM provider API key

    Optional:
        BOXD_ORG           - boxd org to create the machine in (default: the key's org)
        BOXD_REGION        - boxd region, e.g. "fra" or "nyc"
        BOXD_MACHINE_NAME  - machine name (default: "cognee-api")
        LLM_PROVIDER, LLM_MODEL, LLM_ENDPOINT, EMBEDDING_PROVIDER, EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS, EMBEDDING_API_KEY, DEFAULT_USER_PASSWORD
                           - passed to the Cognee server when set

Usage:
    python distributed/deploy/boxd_sandbox.py

Note:
    Machine names are unique, so re-running the script while a previous
    deployment still exists fails with a name conflict. Delete the old machine
    first (use your BOXD_MACHINE_NAME if you set one):
        from boxd import Boxd

        client = Boxd()
        client.machines.delete(client.machines.get("cognee-api").id)
"""

import os
import sys
import time

from boxd import Boxd
from boxd.errors import ConflictError

MACHINE_NAME = "cognee-api"
API_PORT = 8000
APP_DIR = "/home/boxd/cognee"
VENV_PATH = f"{APP_DIR}/.venv"
SERVICE_NAME = "cognee-api"

# Cognee settings passed from the local environment to the server when set.
FORWARDED_ENV = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_ENDPOINT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_API_KEY",
    "DEFAULT_USER_PASSWORD",
)

# The boxd base image ships Python 3.12 and uv, so Cognee goes into a uv venv.
_INSTALL_SCRIPT = (
    "set -e\n"
    f"mkdir -p {APP_DIR}\n"
    f"uv venv -q {VENV_PATH} --python 3.12\n"
    f"VIRTUAL_ENV={VENV_PATH} uv pip install 'cognee[api]'\n"
)

# A systemd service keeps the server running after the exec returns, restarts it
# on failure and starts it again after a reboot. Cognee reads .env from its
# working directory.
_SERVICE_UNIT = f"""[Unit]
Description=Cognee API
Wants=network-online.target
After=network-online.target

[Service]
User=boxd
WorkingDirectory={APP_DIR}
ExecStart={VENV_PATH}/bin/python -m uvicorn cognee.api.client:app --host 0.0.0.0 --port {API_PORT}
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""

_START_SCRIPT = (
    "set -e\n"
    f"sudo install -m 644 {APP_DIR}/{SERVICE_NAME}.service "
    f"/etc/systemd/system/{SERVICE_NAME}.service\n"
    "sudo systemctl daemon-reload\n"
    f"sudo systemctl enable --now {SERVICE_NAME}\n"
)


def run_command(client, machine_id: str, command: str, *, timeout: float, label: str = ""):
    """Run a shell command on the machine, print its output and return the ExecResult."""
    result = client.machines.exec(machine_id, ["bash", "-lc", command], timeout=timeout)

    prefix = f"[{label}] " if label else ""
    for line in result.stdout.splitlines():
        print(f"{prefix}{line}", flush=True)
    for line in result.stderr.splitlines():
        print(f"{prefix}{line}", file=sys.stderr, flush=True)

    return result


def _quote_env_value(name: str, value: str) -> str:
    """Single-quote a value so python-dotenv reads it back unchanged."""
    if "\n" in value or "${" in value:
        raise ValueError(f"{name} must not contain a newline or '${{'")
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_env_file(llm_api_key: str) -> str:
    """Return the .env file for the server: LLM_API_KEY, HOST and the forwarded settings."""
    values = {"LLM_API_KEY": llm_api_key, "HOST": "0.0.0.0"}
    for name in FORWARDED_ENV:
        value = os.environ.get(name)
        if value:
            values[name] = value
    return "".join(f"{name}={_quote_env_value(name, value)}\n" for name, value in values.items())


def wait_for_server_health(client, machine_id: str, retries: int = 30, delay: float = 3.0) -> bool:
    """Poll the Cognee API /health endpoint on the machine until it responds."""
    for attempt in range(retries):
        result = client.machines.exec(
            machine_id,
            ["curl", "-fsS", "--max-time", "10", f"http://localhost:{API_PORT}/health"],
            timeout=30,
        )
        if result.exit_code == 0:
            print("Server is ready!")
            return True
        print(f"  ({attempt + 1}) waiting for server...", flush=True)
        time.sleep(delay)

    run_command(
        client,
        machine_id,
        f"sudo journalctl -u {SERVICE_NAME} -n 30 --no-pager",
        timeout=30,
        label="server-log",
    )
    print("ERROR: Server did not become healthy.", file=sys.stderr)
    return False


def deploy_cognee():
    """Create a boxd machine, start the Cognee API server, and print its URL."""
    if not os.environ.get("BOXD_API_KEY"):
        raise ValueError(
            "BOXD_API_KEY is required. Create one in the boxd console or with "
            "'boxd auth keys create cognee-deploy'."
        )
    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")
    env_file = build_env_file(llm_api_key)

    machine_name = os.environ.get("BOXD_MACHINE_NAME", MACHINE_NAME)

    # The SDK reads BOXD_API_KEY and BOXD_REGION from the environment.
    client = Boxd()

    print("Creating boxd machine for Cognee...")
    try:
        machine = client.machines.create(
            machine_name,
            org=os.environ.get("BOXD_ORG"),
            isolated=True,
            vcpu=2,
            memory="8G",
        )
    except ConflictError as error:
        raise RuntimeError(
            f"Could not create machine '{machine_name}': {error}. If a machine with this name "
            "already exists (likely from a previous run), delete it first, then re-run:\n"
            f'  python -c "from boxd import Boxd; c = Boxd(); '
            f"c.machines.delete(c.machines.get('{machine_name}').id)\""
        ) from error
    print(f"Machine created: {machine.name} ({machine.id})\n")

    try:
        _start_cognee(client, machine.id, env_file)
    except Exception:
        print(
            f"\nDeploy failed. Machine {machine.name} ({machine.id}) is still running, "
            "so you can inspect it. Delete it with:\n"
            f"  python -c \"from boxd import Boxd; Boxd().machines.delete('{machine.id}')\"",
            file=sys.stderr,
        )
        raise

    url = client.machines.get(machine.id).access.url

    print("\nCognee is running!")
    print(f"  Machine: {machine.name} ({machine.id})")
    print(f"\n  API URL: {url}")
    print(f"  Health:  {url}/health")
    print(f"  Docs:    {url}/docs")
    print("\nWith boxd's default idle settings the machine hibernates when idle")
    print("and wakes on the next request.")
    print("To stop the machine:")
    print(f"  python -c \"from boxd import Boxd; Boxd().machines.stop('{machine.id}')\"")
    print("To delete it:")
    print(f"  python -c \"from boxd import Boxd; Boxd().machines.delete('{machine.id}')\"")

    return machine


def _start_cognee(client, machine_id: str, env_file: str) -> None:
    """Install Cognee on a new machine, start the API service and wait until it is healthy."""
    print("Waiting for the machine to be ready...")
    client.machines.wait_until_ready(machine_id, timeout=300)

    print("=== Installing Cognee ===")
    install = run_command(client, machine_id, _INSTALL_SCRIPT, timeout=1800, label="install")
    if install.exit_code != 0:
        raise RuntimeError(f"Failed to install cognee (exit {install.exit_code}): {install.stderr}")
    print()

    print("=== Starting Cognee API server ===")
    # The .env holds the LLM key, so it is created readable by the server user only.
    client.machines.fs.write(machine_id, f"{APP_DIR}/.env", env_file, mode=0o600)
    client.machines.fs.write(machine_id, f"{APP_DIR}/{SERVICE_NAME}.service", _SERVICE_UNIT)
    server_start = run_command(client, machine_id, _START_SCRIPT, timeout=60, label="server")
    if server_start.exit_code != 0:
        raise RuntimeError(
            f"Failed to start cognee API server (exit {server_start.exit_code}): "
            f"{server_start.stderr}"
        )

    print("Waiting for server to start...", flush=True)
    if not wait_for_server_health(client, machine_id):
        raise RuntimeError(
            "Cognee API server did not become healthy; see the server log printed above"
        )

    # API clients (curl, SDKs) should be able to wake the machine from
    # hibernation, so the browser check in front of its URL is turned off.
    client.machines.set_bot_protection(machine_id, mode="off")


if __name__ == "__main__":
    deploy_cognee()
