"""
Cognee on Daytona — deploy Cognee API inside a Daytona sandbox.

Daytona provides secure, isolated cloud sandboxes with persistent volumes.
This script creates a sandbox, installs Cognee, and starts the API server.

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

from daytona import (
    Daytona,
    DaytonaConfig,
    CreateSandboxFromImageParams,
    Image,
    Resources,
)

DAYTONA_API_URL = "https://app.daytona.io/api"


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
    print(f"Sandbox created: {sandbox.id}")

    # Install Cognee and start server in one command to avoid timeout
    # on long pip install. The server starts after install completes.
    print("Installing Cognee and starting API server (this may take a few minutes)...")
    setup_script = (
        "pip install 'cognee[api]' > /tmp/cognee-install.log 2>&1 && "
        "nohup python -m uvicorn cognee.api.client:app "
        "--host 0.0.0.0 --port 8000 > /tmp/cognee-server.log 2>&1 &"
    )
    sandbox.process.exec(f"bash -c \"{setup_script}\"", timeout=600)

    # Generate a signed preview URL (no auth headers needed)
    signed_url = sandbox.create_signed_preview_url(8000, expires_in_seconds=86400)

    print(f"\nCognee sandbox is running!")
    print(f"  Sandbox ID: {sandbox.id}")
    print(f"\n  API URL: {signed_url.url}")
    print(f"  Health:  {signed_url.url}/health")
    print(f"  Docs:    {signed_url.url}/docs")
    print(f"  (URL expires in 24 hours)")
    print(f"\nTo check server logs:")
    print(f"  daytona exec {sandbox.id} -- cat /tmp/cognee-server.log")
    print(f"\nTo stop:")
    print(f"  daytona sandbox stop {sandbox.id}")

    return sandbox


if __name__ == "__main__":
    deploy_cognee()
