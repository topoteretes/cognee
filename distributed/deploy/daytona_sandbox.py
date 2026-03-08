"""
Cognee on Daytona — deploy Cognee API inside a Daytona sandbox.

Daytona provides secure, isolated cloud sandboxes with persistent volumes.
This script creates a sandbox, installs Cognee, and starts the API server,
streaming all output in real time.

Prerequisites:
    pip install daytona

    Set environment variables:
        DAYTONA_API_KEY  — your Daytona API key (from https://app.daytona.io)
        DAYTONA_API_URL  — Daytona API URL (default: https://app.daytona.io/api)
        LLM_API_KEY      — your LLM provider API key

Usage:
    python distributed/deploy/daytona_sandbox.py
"""

import os
import sys
import time
import asyncio

from daytona import (  # type: ignore[import-untyped]
    Daytona,
    DaytonaConfig,
    CreateSandboxFromImageParams,
    SessionExecuteRequest,
    Image,
    Resources,
)

DAYTONA_API_URL = "https://app.daytona.io/api"


def _run_streamed(sandbox, session_id, command, label=""):
    """Run a command in a session and stream stdout/stderr in real time."""
    cmd = sandbox.process.execute_session_command(
        session_id,
        SessionExecuteRequest(command=command, run_async=True),
    )

    prefix = f"[{label}] " if label else ""

    async def _stream():
        await sandbox.process.get_session_command_logs_async(
            session_id,
            cmd.cmd_id,
            lambda stdout: print(f"{prefix}{stdout}", end="", flush=True),
            lambda stderr: print(
                f"{prefix}{stderr}", end="", file=sys.stderr, flush=True
            ),
        )

    asyncio.run(_stream())


def deploy_cognee():
    """Create a Daytona sandbox and start the Cognee API server."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    api_url = os.environ.get("DAYTONA_API_URL", DAYTONA_API_URL)
    llm_api_key = os.environ.get("LLM_API_KEY")

    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    daytona = Daytona(DaytonaConfig(api_key=api_key, api_url=api_url))

    print("Creating Daytona sandbox for Cognee...")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image=Image.debian_slim("3.12"),
            resources=Resources(cpu=2, memory=4, disk=10),
            env_vars={
                "LLM_API_KEY": llm_api_key,
                "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
                "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
                "HOST": "0.0.0.0",
            },
            labels={"app": "cognee", "service": "api"},
        ),
    )
    print(f"Sandbox created: {sandbox.id}\n")

    # Create a session for streamed output
    session_id = "cognee-deploy"
    sandbox.process.create_session(session_id)

    # Step 1: Install Cognee (streamed)
    print("=== Installing Cognee ===")
    _run_streamed(sandbox, session_id, "pip install 'cognee[api]'", label="install")
    print("\n")

    # Step 2: Start the API server as a daemon process using exec (not session)
    # Using a persistent process that survives session cleanup
    print("=== Starting Cognee API server ===")
    sandbox.process.exec(
        "bash -c 'nohup python -m uvicorn cognee.api.client:app "
        "--host 0.0.0.0 --port 8000 > /tmp/cognee-server.log 2>&1 & "
        "echo $!'",
        timeout=10,
    )

    # Wait for server to be ready
    print("Waiting for server to start...", flush=True)
    for i in range(20):
        time.sleep(3)
        result = sandbox.process.exec(
            "python -c \"import urllib.request; "
            "urllib.request.urlopen('http://localhost:8000/health'); "
            "print('OK')\" 2>/dev/null || echo 'WAITING'",
            timeout=10,
        )
        status = result.result.strip()
        if "OK" in status:
            print("Server is ready!")
            break
        print(f"  ({i+1}) {status}", flush=True)
    else:
        # Print server log for debugging
        log = sandbox.process.exec("cat /tmp/cognee-server.log 2>/dev/null", timeout=5)
        print(f"\nServer log:\n{log.result}")
        print("WARNING: Server may not be ready yet. Check logs above.")

    # Generate a signed preview URL (no auth headers needed)
    signed_url = sandbox.create_signed_preview_url(8000, expires_in_seconds=86400)

    print(f"\nCognee is running!")
    print(f"  Sandbox ID: {sandbox.id}")
    print(f"\n  API URL: {signed_url.url}")
    print(f"  Health:  {signed_url.url}/health")
    print(f"  Docs:    {signed_url.url}/docs")
    print(f"  (URL expires in 24 hours)")
    print(f"\nTo check server logs:")
    print(f"  python -c \"from daytona import Daytona, DaytonaConfig; "
          f"d=Daytona(DaytonaConfig(api_key='...', api_url='{api_url}')); "
          f"s=d.get('{sandbox.id}'); print(s.process.exec('cat /tmp/cognee-server.log').result)\"")
    print(f"\nTo stop:")
    print(f"  daytona sandbox stop {sandbox.id}")

    return sandbox


if __name__ == "__main__":
    deploy_cognee()
