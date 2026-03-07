"""
Cognee on Daytona — deploy Cognee API inside a Daytona sandbox.

Daytona provides secure, isolated cloud sandboxes with persistent volumes.
This script creates a sandbox, installs Cognee, and starts the API server.

Prerequisites:
    pip install daytona

    Set environment variables:
        DAYTONA_API_KEY  — your Daytona API key (from https://app.daytona.io)
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


def deploy_cognee():
    """Create a Daytona sandbox and start the Cognee API server."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    llm_api_key = os.environ.get("LLM_API_KEY")

    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    daytona = Daytona(DaytonaConfig(api_key=api_key))

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

    # Install Cognee
    print("Installing Cognee...")
    sandbox.process.exec("pip install 'cognee[api]'")

    # Start the API server
    print("Starting Cognee API server...")
    response = sandbox.process.exec(
        "nohup python -m uvicorn cognee.api.client:app "
        "--host 0.0.0.0 --port 8000 &"
    )
    print(response.result)

    print(f"\nCognee sandbox is running!")
    print(f"Sandbox ID: {sandbox.id}")
    print("Use 'daytona preview <sandbox-id> 8000' to access the API")

    return sandbox


if __name__ == "__main__":
    deploy_cognee()
