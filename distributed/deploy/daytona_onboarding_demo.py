"""
Claude Code + Cognee on Daytona — Shared Memory Demo

Spins up one Cognee sandbox (shared knowledge graph = public memory) and N
Claude Code agent sandboxes. Each agent runs the cognee-memory plugin from
cognee-integrations, which:

  * captures tool calls into a per-directory session cache (private memory)
  * auto-injects relevant context on every UserPromptSubmit
  * bridges session data into the permanent graph on SessionEnd

Agents run sequentially so later agents can recall what earlier agents
deposited into the shared graph. Each agent's session cache stays private
to its own sandbox.

Prerequisites:
    pip install daytona

    Set environment variables:
        DAYTONA_API_KEY     — https://app.daytona.io
        ANTHROPIC_API_KEY   — for Claude Code itself
        LLM_API_KEY         — for Cognee (graph extraction, search)

Usage:
    python distributed/deploy/daytona_onboarding_demo.py \\
        --repo https://github.com/topoteretes/cognee

    # Reuse an existing Cognee sandbox:
    python distributed/deploy/daytona_onboarding_demo.py \\
        --cognee-url https://<id>-8000.daytona.work
"""

import argparse
import os
import sys
import time
import asyncio
import shlex

from daytona import (
    Daytona,
    DaytonaConfig,
    DaytonaAuthorizationError,
    CreateSandboxFromImageParams,
    SessionExecuteRequest,
    Image,
    Resources,
)

DAYTONA_API_URL = "https://app.daytona.io/api"
DEFAULT_REPO = "https://github.com/topoteretes/cognee"
INTEGRATIONS_REPO = "https://github.com/topoteretes/cognee-integrations"
PLUGIN_SUBPATH = "integrations/claude-code"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_daytona():
    api_key = os.environ.get("DAYTONA_API_KEY")
    api_url = os.environ.get("DAYTONA_API_URL", DAYTONA_API_URL)
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required")
    return Daytona(DaytonaConfig(api_key=api_key, api_url=api_url))


def _run_streamed(sandbox, session_id, command, label="", timeout=600):
    """Run a command in a sandbox session, falling back to blocking exec on WS errors."""
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
        f"import urllib.request\n"
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


def _print_tail(result, label, lines=5):
    """Print the last N lines of a command result."""
    if not result or not result.result:
        return
    for line in result.result.strip().splitlines()[-lines:]:
        print(f"  [{label}] {line}", flush=True)


# ---------------------------------------------------------------------------
# Sandbox 1: Cognee API (shared public memory)
# ---------------------------------------------------------------------------

def deploy_cognee_sandbox():
    """Deploy the Cognee API server. Returns (sandbox, signed_url)."""
    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    daytona = _get_daytona()

    print("=== Creating Cognee sandbox (shared public memory) ===")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image=Image.debian_slim("3.12"),
            resources=Resources(cpu=2, memory=4, disk=10),
            # Default is 15 min — Daytona idle-stopped the sandbox mid-demo
            # last run, which made the plugin fall back to local storage.
            # 0 disables auto-stop entirely.
            auto_stop_interval=0,
            env_vars={
                "LLM_API_KEY": llm_api_key,
                "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
                "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
                "HOST": "0.0.0.0",
                "CACHING": "true",
                "REQUIRE_AUTHENTICATION": "False",
                "ENABLE_BACKEND_ACCESS_CONTROL": "False",
            },
            labels={"app": "cognee", "service": "api", "demo": "shared-memory"},
        ),
    )
    print(f"  Sandbox ID: {sandbox.id}")

    print("  Installing cognee (this may take a few minutes)...")
    result = sandbox.process.exec("pip install 'cognee[api]'", timeout=600)
    _print_tail(result, "cognee")

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

    # Fail fast if auth wasn't actually disabled. The first pass of this demo
    # spun up Cognee with auth still on (env var was added later); agents
    # then fell back to local storage silently.
    import urllib.request
    try:
        with urllib.request.urlopen(f"{signed_url.url}/api/v1/datasets", timeout=10) as r:
            if r.status != 200:
                raise RuntimeError(f"unexpected status {r.status}")
    except Exception as e:
        raise RuntimeError(
            f"Cognee /api/v1/datasets did not accept unauthenticated requests "
            f"(REQUIRE_AUTHENTICATION not picked up?): {e}"
        )
    print("  Auth check: /api/v1/datasets accepts unauth'd requests.")
    return sandbox, signed_url.url


def _remote_has_data(cognee_url):
    """Poll Cognee for evidence that an agent actually wrote to the backend."""
    import urllib.request
    import json
    try:
        with urllib.request.urlopen(f"{cognee_url}/api/v1/datasets", timeout=10) as r:
            data = json.loads(r.read())
        return bool(data), data
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Sandbox N: Claude Code agent (private per-agent memory + shared graph)
# ---------------------------------------------------------------------------

def _create_with_quota_retry(daytona, params, retries=6, delay=20):
    """Create a sandbox, retrying on the Daytona memory-quota error.

    Daytona's delete is async: the concurrent-memory quota takes a short
    while to release after a sandbox is destroyed. When we tear down one
    agent and immediately create the next, the first attempt can still hit
    "Total memory limit exceeded" even though we're under the cap.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return daytona.create(params)
        except DaytonaAuthorizationError as e:
            last_error = e
            if "memory limit" not in str(e).lower():
                raise
            print(f"  quota not yet released, retrying in {delay}s (attempt {attempt + 1}/{retries})...")
            time.sleep(delay)
    raise last_error if last_error else RuntimeError("sandbox create failed")


def deploy_claude_code_agent(cognee_url, target_repo, label):
    """Deploy a Claude Code sandbox with the cognee-memory plugin.

    Each agent gets:
      * its own Daytona sandbox (isolated filesystem, process space)
      * a per-directory session ID (private session cache in Cognee)
      * the shared Cognee backend at `cognee_url` (public graph)
    """
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    daytona = _get_daytona()

    print(f"\n=== Creating Claude Code agent sandbox: {label} ===")
    sandbox = _create_with_quota_retry(daytona,
        CreateSandboxFromImageParams(
            image=Image.debian_slim("3.12"),
            # 4 GiB was killed by the OOM killer when Claude Code + the
            # cognee plugin + the cognee SDK all loaded at once. 6 GiB leaves
            # the Cognee sandbox (4 GiB) exactly at the 10 GiB tier cap.
            resources=Resources(cpu=2, memory=6, disk=10),
            env_vars={
                "ANTHROPIC_API_KEY": anthropic_api_key,
                "LLM_API_KEY": llm_api_key,
                "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
                "COGNEE_SERVICE_URL": cognee_url,
                "COGNEE_PLUGIN_DATASET": "shared_codebase_memory",
                "COGNEE_SESSION_STRATEGY": "per-directory",
                "COGNEE_SESSION_PREFIX": label,
                "CACHING": "true",
            },
            labels={"app": "claude-code", "agent": label, "demo": "shared-memory"},
        ),
    )
    print(f"  Sandbox ID: {sandbox.id}")

    session_id = "agent-setup"
    sandbox.process.create_session(session_id)

    print("  Installing prerequisites...")
    sandbox.process.exec(
        "apt-get update -qq && apt-get install -y -qq curl git ca-certificates xz-utils",
        timeout=180,
    )

    # Node.js 22 via binary tarball — apt only ships Node 18.
    print("  Installing Node.js 22...")
    node_version = "v22.16.0"
    sandbox.process.exec(
        f"curl -fsSL https://nodejs.org/dist/{node_version}/node-{node_version}-linux-x64.tar.xz "
        f"| tar -xJ -C /usr/local --strip-components=1",
        timeout=180,
    )
    result = sandbox.process.exec("node --version && npm --version", timeout=10)
    node_info = result.result.strip()
    print(f"  {node_info}")
    if not node_info.startswith("v22"):
        raise RuntimeError(f"Expected Node 22 but got: {node_info}")

    print("  Installing Claude Code CLI...")
    result = sandbox.process.exec(
        "npm install -g @anthropic-ai/claude-code 2>&1", timeout=600
    )
    _print_tail(result, "claude")

    # Plugin depends on the cognee Python SDK for its hooks/skills.
    print("  Installing cognee SDK (plugin dependency)...")
    result = sandbox.process.exec("pip install cognee", timeout=600)
    _print_tail(result, "cognee-sdk")

    print("  Cloning cognee-integrations (for the plugin)...")
    sandbox.process.exec(
        f"git clone --depth 1 {INTEGRATIONS_REPO} /opt/cognee-integrations 2>&1",
        timeout=60,
    )

    print(f"  Cloning target repo {target_repo}...")
    _run_streamed(
        sandbox, session_id,
        f"git clone --depth 1 {target_repo} /workspace 2>&1",
        label="git",
    )

    # Claude Code refuses --dangerously-skip-permissions when running as root.
    # Create an unprivileged user and dump the sandbox env so it can source
    # the provider keys + plugin config when we invoke claude as that user.
    print("  Creating agent user...")
    sandbox.process.exec(
        "useradd -m -s /bin/bash agent && "
        "chown -R agent:agent /workspace /opt/cognee-integrations && "
        "env | grep -E '^(ANTHROPIC_API_KEY|LLM_|COGNEE_|CACHING)=' "
        "| sed 's/^/export /' > /home/agent/env && "
        "chown agent:agent /home/agent/env && chmod 600 /home/agent/env",
        timeout=15,
    )

    print(f"  Agent sandbox {label} ready.\n")
    return sandbox


# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------

def run_agent_task(sandbox, label, prompt):
    """Run one Claude Code task with the cognee-memory plugin loaded.

    Uses `claude -p` (print mode) for a single non-interactive turn.
    The plugin:
      * searches session cache + shared graph on UserPromptSubmit (auto)
      * captures tool calls into the session cache on PostToolUse (auto)
      * bridges the session into the permanent graph on SessionEnd (auto)
    """
    print(f"\n{'=' * 60}")
    print(f"  AGENT:  {label}")
    print(f"  PROMPT: {prompt[:80]}...")
    print(f"{'=' * 60}\n")

    plugin_dir = f"/opt/cognee-integrations/{PLUGIN_SUBPATH}"
    inner = (
        f"set -a && . /home/agent/env && set +a && "
        f"cd /workspace && "
        f"claude --plugin-dir {shlex.quote(plugin_dir)} "
        f"--dangerously-skip-permissions "
        f"-p {shlex.quote(prompt)} < /dev/null"
    )
    cmd = f"runuser -u agent -- bash -c {shlex.quote(inner)}"

    session_name = f"task-{label}"
    sandbox.process.create_session(session_name)
    _run_streamed(sandbox, session_name, cmd, label=label, timeout=1800)
    print(f"\n  [{label}] complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code + Cognee on Daytona: shared public memory demo"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("TARGET_REPO", DEFAULT_REPO),
        help="Git URL of the repository the agents will explore",
    )
    parser.add_argument(
        "--cognee-url",
        default=os.environ.get("COGNEE_SERVICE_URL"),
        help="Reuse an existing Cognee API URL instead of deploying a new sandbox",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CLAUDE CODE + COGNEE SHARED MEMORY DEMO")
    print(f"  Target repo: {args.repo}")
    print("=" * 60 + "\n")

    cognee_sandbox = None
    if args.cognee_url:
        cognee_url = args.cognee_url
        print(f"  Reusing existing Cognee API: {cognee_url}\n")
    else:
        cognee_sandbox, cognee_url = deploy_cognee_sandbox()

    # Three agents, three roles. Each has private per-directory session memory;
    # all share the Cognee backend for the permanent graph.
    agents = [
        (
            "arch",
            "Explore this codebase. Map the project structure, key entry "
            "points, main modules, and overall architecture. Use "
            "/cognee-memory:cognee-remember to store the most important "
            "findings in the shared knowledge graph under the 'project' "
            "category so other agents can read them.",
        ),
        (
            "api",
            "Before exploring, use /cognee-memory:cognee-search to recall "
            "anything the architecture agent stored in the shared graph. "
            "Then describe the API layer — routes, middleware, auth, request "
            "handling. Store your findings with /cognee-memory:cognee-remember.",
        ),
        (
            "tests",
            "Use /cognee-memory:cognee-search to pull in what the architecture "
            "and API agents already learned. Then describe the testing strategy — "
            "which suites exist, what they cover, and any gaps. Store findings "
            "with /cognee-memory:cognee-remember.",
        ),
    ]

    # Run agents one at a time and tear each down before the next: Daytona's
    # default tier caps total concurrent memory at 10 GiB (Cognee = 4 GiB,
    # each agent = 4 GiB). Personal session memory already lives on the
    # Cognee backend, so destroying the sandbox doesn't lose anything.
    daytona = _get_daytona()
    ran = []
    for label, prompt in agents:
        sandbox = deploy_claude_code_agent(cognee_url, args.repo, label)
        try:
            run_agent_task(sandbox, label, prompt)
            ran.append((label, sandbox.id))
        finally:
            try:
                daytona.delete(sandbox)
                print(f"  [{label}] sandbox {sandbox.id} destroyed.")
            except Exception as e:
                print(f"  [{label}] WARNING: failed to delete sandbox: {e}")

        # Fail fast if nothing reached the shared backend — previously the
        # plugin fell back to local SDK mode silently, so all three agents
        # "succeeded" while the remote stayed empty.
        has_data, payload = _remote_has_data(cognee_url)
        print(f"  Remote check after {label}: datasets={payload}")
        if not has_data and label == "arch":
            raise RuntimeError(
                "Cognee backend still has no datasets after arch — plugin did "
                "not reach the remote. Aborting before burning agent credits."
            )

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print(f"  Cognee API:      {cognee_url}")
    if cognee_sandbox:
        print(f"  Cognee sandbox:  {cognee_sandbox.id}")
    for label, sandbox_id in ran:
        print(f"  Agent {label:<6} ran in: {sandbox_id} (destroyed)")
    print(f"  Datasets:        {cognee_url}/api/v1/datasets")
    print(f"  Swagger:         {cognee_url}/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
