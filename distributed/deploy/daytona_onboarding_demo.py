"""
OpenClaw + Cognee on Daytona — Codebase Onboarding Demo

Spins up two Daytona sandboxes:
  1. Cognee API server (memory backend)
  2. OpenClaw agent with cognee-openclaw plugin (the onboarding agent)

The agent explores a target repository across multiple sessions. Between
sessions the agent process is killed to simulate a crash. Because memory
lives in Cognee (running in its own sandbox), the agent picks up where it
left off — no context is lost.

Prerequisites:
    pip install daytona

    Set environment variables:
        DAYTONA_API_KEY  — your Daytona API key (from https://app.daytona.io)
        LLM_API_KEY      — your LLM provider API key

Usage:
    python distributed/deploy/daytona_onboarding_demo.py \\
        --repo https://github.com/topoteretes/cognee

    # Or use defaults (onboards this repo):
    python distributed/deploy/daytona_onboarding_demo.py
"""

import argparse
import os
import sys
import time
import asyncio
import textwrap

from daytona_sdk import (
    Daytona,
    DaytonaConfig,
    CreateSandboxFromImageParams,
    SessionExecuteRequest,
    Image,
    Resources,
)

DAYTONA_API_URL = "https://app.daytona.io/api"
DEFAULT_REPO = "https://github.com/topoteretes/cognee"

OPENCLAW_CONFIG_TEMPLATE = textwrap.dedent("""\
    plugins:
      entries:
        cognee-openclaw:
          enabled: true
          config:
            baseUrl: "{cognee_url}"
            datasetName: "codebase-onboarding"
            searchType: "GRAPH_COMPLETION"
            autoRecall: true
            autoIndex: true
            autoCognify: true
            maxResults: 10
            maxTokens: 1024
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_daytona():
    api_key = os.environ.get("DAYTONA_API_KEY")
    api_url = os.environ.get("DAYTONA_API_URL", DAYTONA_API_URL)
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required")
    return Daytona(DaytonaConfig(api_key=api_key, api_url=api_url))


def _run_streamed(sandbox, session_id, command, label="", timeout=300):
    """Run a command in a sandbox session, falling back to non-streaming on WS errors."""
    prefix = f"[{label}] " if label else ""
    try:
        cmd = sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=command, run_async=True),
        )

        async def _stream():
            await sandbox.process.get_session_command_logs_async(
                session_id,
                cmd.cmd_id,
                lambda stdout: print(f"{prefix}{stdout}", end="", flush=True),
                lambda stderr: print(f"{prefix}{stderr}", end="", file=sys.stderr, flush=True),
            )

        asyncio.run(_stream())
    except Exception as e:
        print(f"\n  {prefix}Streaming failed ({e}), falling back to blocking exec...")
        result = sandbox.process.exec(command, timeout=timeout)
        if result.result:
            for line in result.result.splitlines():
                print(f"{prefix}{line}", flush=True)


def _wait_for_health(sandbox, port=8000, retries=30, interval=3):
    """Poll a health endpoint until it responds."""
    sandbox.process.exec(
        "cat > /tmp/healthcheck.py << 'PYEOF'\n"
        f"import urllib.request, sys\n"
        "try:\n"
        f"    urllib.request.urlopen('http://localhost:{port}/health', timeout=5)\n"
        "    print('OK')\n"
        "except Exception:\n"
        "    print('WAITING')\n"
        "PYEOF",
        timeout=5,
    )
    for i in range(retries):
        time.sleep(interval)
        try:
            result = sandbox.process.exec("python /tmp/healthcheck.py", timeout=10)
            if "OK" in result.result.strip():
                return True
        except Exception:
            pass
        print(f"  ({i + 1}) waiting...", flush=True)
    return False


# ---------------------------------------------------------------------------
# Sandbox 1: Cognee API
# ---------------------------------------------------------------------------

def deploy_cognee_sandbox():
    """Deploy the Cognee API server in a Daytona sandbox.

    Returns:
        (sandbox, signed_url) — the sandbox object and its public API URL.
    """
    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    daytona = _get_daytona()

    print("=== Creating Cognee sandbox ===")
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
            labels={"app": "cognee", "service": "api", "demo": "onboarding"},
        ),
    )
    print(f"  Sandbox ID: {sandbox.id}")

    # Install Cognee (blocking exec — pip install is too slow for WS streaming)
    print("  Installing cognee (this may take a few minutes)...")
    result = sandbox.process.exec("pip install 'cognee[api]'", timeout=600)
    if result.result:
        # Print just the last few lines (success/error summary)
        lines = result.result.strip().splitlines()
        for line in lines[-5:]:
            print(f"  [cognee] {line}", flush=True)

    # Start API server
    print("\n  Starting Cognee API server...")
    sandbox.process.exec(
        "nohup python -m uvicorn cognee.api.client:app "
        "--host 0.0.0.0 --port 8000 > /tmp/cognee-server.log 2>&1 &",
        timeout=10,
    )

    print("  Waiting for server...", flush=True)
    if not _wait_for_health(sandbox, retries=40, interval=5):
        try:
            log = sandbox.process.exec("tail -30 /tmp/cognee-server.log", timeout=5)
            print(f"\n  Server log:\n{log.result}")
        except Exception:
            pass
        print("  WARNING: Cognee server may not be ready.")

    signed_url = sandbox.create_signed_preview_url(8000, expires_in_seconds=86400)
    print(f"  Cognee API ready: {signed_url.url}")
    return sandbox, signed_url.url


# ---------------------------------------------------------------------------
# Sandbox 2: OpenClaw Agent
# ---------------------------------------------------------------------------

def deploy_openclaw_sandbox(cognee_url, target_repo):
    """Deploy an OpenClaw agent in a Daytona sandbox with cognee-openclaw plugin.

    Args:
        cognee_url: Public URL of the Cognee API sandbox.
        target_repo: Git URL of the repo the agent will onboard onto.

    Returns:
        The sandbox object.
    """
    daytona = _get_daytona()

    print("\n=== Creating OpenClaw agent sandbox ===")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image=Image.debian_slim("3.12"),
            resources=Resources(cpu=2, memory=4, disk=10),
            env_vars={
                "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
                "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
                "COGNEE_API_URL": cognee_url,
            },
            labels={"app": "openclaw", "service": "agent", "demo": "onboarding"},
        ),
    )
    print(f"  Sandbox ID: {sandbox.id}")

    session_id = "agent-setup"
    sandbox.process.create_session(session_id)

    # Install prerequisites + Node.js 22 via binary tarball
    # (OpenClaw requires Node >= 22.14.0; apt only has Node 18)
    print("  Installing prerequisites...")
    sandbox.process.exec(
        "apt-get update -qq && apt-get install -y -qq curl git ca-certificates xz-utils",
        timeout=120,
    )

    print("  Installing Node.js 22...")
    node_version = "v22.16.0"
    sandbox.process.exec(
        f"curl -fsSL https://nodejs.org/dist/{node_version}/node-{node_version}-linux-x64.tar.xz "
        f"| tar -xJ -C /usr/local --strip-components=1",
        timeout=120,
    )

    result = sandbox.process.exec("node --version && npm --version", timeout=10)
    node_info = result.result.strip()
    print(f"  {node_info}")
    if not node_info.startswith("v22"):
        raise RuntimeError(f"Expected Node 22 but got: {node_info}")

    # Install OpenClaw CLI + plugin (blocking exec — npm install is too slow for WS streaming)
    print("  Installing OpenClaw + cognee plugin (this may take a few minutes)...")
    result = sandbox.process.exec(
        "npm install -g openclaw@latest @cognee/cognee-openclaw 2>&1",
        timeout=600,
    )
    if result.result:
        lines = result.result.strip().splitlines()
        for line in lines[-5:]:
            print(f"  [openclaw] {line}", flush=True)

    # Run onboard to install daemon and resolve dependencies
    print("  Running OpenClaw onboard...")
    result = sandbox.process.exec("openclaw onboard --install-daemon 2>&1", timeout=120)
    if result.result:
        lines = result.result.strip().splitlines()
        for line in lines[-5:]:
            print(f"  [openclaw] {line}", flush=True)

    # Write config
    config_content = OPENCLAW_CONFIG_TEMPLATE.format(cognee_url=cognee_url)
    sandbox.process.exec(
        f"mkdir -p ~/.openclaw && cat > ~/.openclaw/config.yaml << 'CFGEOF'\n"
        f"{config_content}"
        f"CFGEOF",
        timeout=5,
    )
    print("\n  Plugin configured.")

    # Clone target repo
    print(f"  Cloning {target_repo}...")
    _run_streamed(
        sandbox, session_id,
        f"git clone --depth 1 {target_repo} /workspace 2>&1",
        label="git",
    )

    # Create memory directory
    sandbox.process.exec("mkdir -p /workspace/memory", timeout=5)
    print("\n  Agent sandbox ready.")
    return sandbox


# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------

def run_onboarding_session(sandbox, session_name, prompt):
    """Run one OpenClaw agent session with a given prompt.

    The cognee-openclaw plugin auto-recalls relevant memories before the
    prompt runs and auto-indexes new memory files after it completes.
    """
    print(f"\n{'=' * 60}")
    print(f"  SESSION: {session_name}")
    print(f"  PROMPT:  {prompt[:80]}...")
    print(f"{'=' * 60}\n")

    sandbox.process.create_session(session_name)
    _run_streamed(
        sandbox, session_name,
        f'cd /workspace && openclaw run --prompt "{prompt}"',
        label=session_name,
    )
    print(f"\n  [{session_name}] complete.\n")


def simulate_restart(sandbox):
    """Kill agent processes to simulate a crash."""
    print("\n" + "~" * 60)
    print("  SIMULATING AGENT CRASH / RESTART")
    print("  (killing agent process, memory persists in Cognee)")
    print("~" * 60 + "\n")
    try:
        sandbox.process.exec("pkill -f openclaw || true", timeout=5)
    except Exception:
        pass
    time.sleep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw + Cognee on Daytona: Codebase Onboarding Demo"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("TARGET_REPO", DEFAULT_REPO),
        help="Git URL of the repository to onboard (default: cognee itself)",
    )
    parser.add_argument(
        "--cognee-url",
        default=os.environ.get("COGNEE_API_URL"),
        help="Reuse an existing Cognee API URL instead of deploying a new sandbox",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CODEBASE ONBOARDING DEMO")
    print(f"  Target repo: {args.repo}")
    print("=" * 60 + "\n")

    # Deploy or reuse Cognee sandbox
    cognee_sandbox = None
    if args.cognee_url:
        cognee_url = args.cognee_url
        print(f"  Reusing existing Cognee API: {cognee_url}\n")
    else:
        cognee_sandbox, cognee_url = deploy_cognee_sandbox()

    agent_sandbox = deploy_openclaw_sandbox(cognee_url, args.repo)

    # --- Session 1: Architecture Discovery ---
    run_onboarding_session(
        agent_sandbox, "session-1",
        "Explore this codebase. Understand the project structure, "
        "key entry points, main modules, and overall architecture. "
        "Write your findings to memory/architecture.md",
    )

    # --- Simulate crash ---
    simulate_restart(agent_sandbox)

    # --- Session 2: API Deep Dive (should recall Session 1) ---
    run_onboarding_session(
        agent_sandbox, "session-2",
        "What is the API layer structure? Explore routes, middleware, "
        "authentication, and request handling patterns. "
        "Write your findings to memory/api-layer.md",
    )

    # --- Session 3: Cross-cutting Concerns (should recall 1+2) ---
    run_onboarding_session(
        agent_sandbox, "session-3",
        "What error handling and logging patterns are used across "
        "the codebase? Are they consistent? "
        "Write your findings to memory/error-handling.md",
    )

    # Summary
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print(f"  Cognee API:      {cognee_url}")
    if cognee_sandbox:
        print(f"  Cognee sandbox:  {cognee_sandbox.id}")
    print(f"  Agent sandbox:   {agent_sandbox.id}")
    print(f"  Datasets:        {cognee_url}/api/v1/datasets")
    print(f"  Swagger:         {cognee_url}/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
