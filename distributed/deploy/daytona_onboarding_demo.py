"""
Claude Code + Cognee + Moss on Daytona — Shared Memory Demo (Snapshot Edition)

Why no separate Cognee sandbox:
    Daytona's public preview URL is served through a proxy that rejects
    sandbox-to-sandbox traffic (Connection reset by peer), even with
    public=True. A previous version of this script deployed Cognee in
    its own sandbox; every agent silently fell back to local-SDK mode
    and "succeeded" against an ephemeral local DB, so the shared graph
    stayed empty across runs.

Why no direct volume-mounted DBs either:
    Daytona volumes are mountpoint-s3 FUSE mounts. SQLite, Kuzu, and
    LanceDB all need random writes + file locking, which mountpoint-s3
    explicitly doesn't support ("disk I/O error" on CREATE TABLE). The
    Daytona docs confirm: volumes are not suitable for block-level
    database access.

What works instead:
    The volume is used as a snapshot bucket. Each agent runs cognee
    against a LOCAL disk path. Vectors are stored in and queried from
    Moss (cloud index) in-process via cognee-community-vector-adapter-moss. Before the agent starts, we extract the
    latest shared snapshot from the volume into the local state dir.
    After the agent finishes, we tar local state back into the volume
    as a single object. Because agents run sequentially, there are no
    concurrent writes, and whole-object S3 writes are exactly what
    mountpoint-s3 is designed for.

Prerequisites:
    pip install daytona

    Set environment variables:
        DAYTONA_API_KEY     - https://app.daytona.io
        ANTHROPIC_API_KEY   - for Claude Code itself
        LLM_API_KEY         - for Cognee (graph extraction + search)
        MOSS_PROJECT_ID     - https://moss.dev
        MOSS_PROJECT_KEY    - https://moss.dev

    Optional:
        COGNEE_INSTALL_SPEC - Cognee package, wheel, or git URL to install
                              in each sandbox. Defaults to the hackathon
                              dev release pinned below.

Usage:
    python distributed/deploy/daytona_onboarding_demo.py \\
        --repo https://github.com/topoteretes/cognee \\
        --keep-volume
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
    VolumeMount,
)

DAYTONA_API_URL = "https://app.daytona.io/api"
DEFAULT_REPO = "https://github.com/topoteretes/cognee"
INTEGRATIONS_REPO = "https://github.com/topoteretes/cognee-integrations"
PLUGIN_SUBPATH = "integrations/claude-code"
DEFAULT_COGNEE_INSTALL_SPEC = "cognee==1.0.4.dev0"
MOSS_ADAPTER_INSTALL_SPEC = "cognee-community-vector-adapter-moss==0.1.1"

VOLUME_NAME = "cognee-shared-memory"
MOUNT_PATH = "/shared-cognee"
# Path inside each agent sandbox where cognee's local DBs live. This is on
# the sandbox's real filesystem (block storage), so SQLite/Kuzu work
# correctly. Vectors live in Moss; snapshots of this dir are synced to/from the volume.
LOCAL_STATE = "/var/cognee-state"
SNAPSHOT_FILE = "state.tar.gz"


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


def _print_tail(result, label, lines=5):
    """Print the last N lines of a command result."""
    if not result or not result.result:
        return
    for line in result.result.strip().splitlines()[-lines:]:
        print(f"  [{label}] {line}", flush=True)


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
            print(
                f"  quota not yet released, retrying in {delay}s (attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)
    raise last_error if last_error else RuntimeError("sandbox create failed")


# ---------------------------------------------------------------------------
# Shared volume (replaces the Cognee service sandbox)
# ---------------------------------------------------------------------------


def ensure_shared_volume(daytona, retries=30, delay=2):
    """Create (or reuse) the Daytona volume used as Cognee's shared storage.

    Daytona creates volumes asynchronously; a newly-created volume sits in
    `pending_create` for a few seconds before becoming `ready`. Mounting
    it before then fails the sandbox create with a validation error, so
    poll until it's ready.
    """
    print("=== Preparing shared volume ===")
    vol = daytona.volume.get(VOLUME_NAME, create=True)
    print(f"  Volume: {vol.name} ({vol.id}), state={vol.state}")
    for _ in range(retries):
        if str(vol.state).upper().endswith("READY"):
            return vol
        time.sleep(delay)
        vol = daytona.volume.get(VOLUME_NAME, create=False)
        print(f"  ...waiting, state={vol.state}")
    raise RuntimeError(f"Volume {VOLUME_NAME} did not become ready: {vol.state}")


# ---------------------------------------------------------------------------
# Claude Code agent sandbox
# ---------------------------------------------------------------------------


def deploy_claude_code_agent(volume_id, target_repo, label, cognee_install_spec):
    """Deploy a Claude Code sandbox that shares Cognee storage via volume mount.

    Each agent runs Cognee in-process. DATA/SYSTEM/CACHE_ROOT_DIRECTORY point
    at local sandbox disk so SQLite/Kuzu can use file locking. A tar snapshot
    of that local state is synced through the mounted volume between agents.
    Vectors are stored in and queried from Moss.
    """
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    moss_project_id = os.environ.get("MOSS_PROJECT_ID")
    if not moss_project_id:
        raise ValueError("MOSS_PROJECT_ID environment variable is required")

    moss_project_key = os.environ.get("MOSS_PROJECT_KEY")
    if not moss_project_key:
        raise ValueError("MOSS_PROJECT_KEY environment variable is required")

    daytona = _get_daytona()

    print(f"\n=== Creating Claude Code agent sandbox: {label} ===")
    sandbox = _create_with_quota_retry(
        daytona,
        CreateSandboxFromImageParams(
            image=Image.base("python:3.12-slim-trixie"),
            # 4 GiB was OOM-killed during claude+plugin+cognee SDK startup.
            # 6 GiB leaves us at the 10 GiB tier cap with nothing else.
            resources=Resources(cpu=2, memory=6, disk=10),
            volumes=[VolumeMount(volume_id=volume_id, mount_path=MOUNT_PATH)],
            env_vars={
                "ANTHROPIC_API_KEY": anthropic_api_key,
                "LLM_API_KEY": llm_api_key,
                "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
                "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
                # Tell Cognee to use Moss as its vector DB.
                "VECTOR_DB_PROVIDER": "moss",
                "VECTOR_DB_KEY": moss_project_key,
                "VECTOR_DB_NAME": moss_project_id,
                "VECTOR_DATASET_DATABASE_HANDLER": "moss",
                # Cognee local storage — on the sandbox's real filesystem,
                # NOT on the volume (mountpoint-s3 can't host SQLite/Kuzu).
                # Synced to the volume before/after each agent's run.
                "DATA_ROOT_DIRECTORY": f"{LOCAL_STATE}/data",
                "SYSTEM_ROOT_DIRECTORY": f"{LOCAL_STATE}/system",
                "CACHE_ROOT_DIRECTORY": f"{LOCAL_STATE}/cache",
                # Plugin config. No COGNEE_SERVICE_URL -> plugin uses local mode.
                "COGNEE_PLUGIN_DATASET": "shared_codebase_memory",
                "COGNEE_SESSION_STRATEGY": "per-directory",
                "COGNEE_SESSION_PREFIX": label,
                "CACHING": "true",
                "COGNEE_INSTALL_SPEC": cognee_install_spec,
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

    # Node.js 22 via tarball — apt only ships Node 18, Claude Code needs >=22.
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
    result = sandbox.process.exec("npm install -g @anthropic-ai/claude-code 2>&1", timeout=600)
    _print_tail(result, "claude")

    # Plugin depends on the cognee Python SDK for its hooks/skills.
    print(f"  Installing cognee SDK ({cognee_install_spec})...")
    install_cmd = (
        f"pip install {shlex.quote(cognee_install_spec)} {shlex.quote(MOSS_ADAPTER_INSTALL_SPEC)}"
    )
    result = sandbox.process.exec(
        install_cmd,
        timeout=600,
    )
    _print_tail(result, "cognee-sdk")

    # Write a sitecustomize.py so the Moss adapter auto-registers for every
    # Python process in the sandbox, including the hook subprocesses fired
    # by the cognee-memory plugin. sitecustomize is imported by Python before
    # any user code, making this the most reliable registration point.
    print("  Registering Moss adapter for all Python processes...")
    site_packages = sandbox.process.exec(
        'python -c "import site; print(site.getsitepackages()[0])"',
        timeout=10,
    ).result.strip()
    sandbox.process.exec(
        f"cat > {site_packages}/sitecustomize.py << 'EOF'\n"
        "import os\n"
        "import sys\n"
        "try:\n"
        "    # Importing this submodule registers the adapter.\n"
        "    from cognee_community_vector_adapter_moss import register  # noqa: F401\n"
        "    # Programmatic config (matches the PyPI docs exactly). Env vars\n"
        "    # alone are not honored by Cognee at the plugin's config path.\n"
        "    from cognee import config\n"
        "    config.set_vector_db_config({\n"
        "        'vector_db_provider': 'moss',\n"
        "        'vector_db_key': os.getenv('VECTOR_DB_KEY'),\n"
        "        'vector_db_name': os.getenv('VECTOR_DB_NAME'),\n"
        "        'vector_dataset_database_handler': 'moss',\n"
        "    })\n"
        "except Exception as e:\n"
        "    print(f'[moss-adapter] setup failed: {e!r}', file=sys.stderr)\n"
        "EOF",
        timeout=5,
    )

    print("  Cloning cognee-integrations (for the plugin)...")
    sandbox.process.exec(
        f"git clone --depth 1 {INTEGRATIONS_REPO} /opt/cognee-integrations 2>&1",
        timeout=60,
    )

    # Bump the plugin's SessionStart hook timeout. The default (15s) is
    # shorter than a cognee cold import on first run.
    sandbox.process.exec(
        r"sed -i 's/\"timeout\": 15/\"timeout\": 90/g' "
        "/opt/cognee-integrations/integrations/claude-code/hooks/hooks.json",
        timeout=5,
    )

    print(f"  Cloning target repo {target_repo}...")
    _run_streamed(
        sandbox,
        session_id,
        f"git clone --depth 1 {target_repo} /workspace 2>&1",
        label="git",
    )

    # Claude Code refuses --dangerously-skip-permissions when running as root.
    # Create the agent user and its local state dir (cognee DB lives here,
    # on real block storage — see module docstring). We DO NOT chown the
    # volume mount — it's mountpoint-s3 FUSE and chown fails "Operation
    # not permitted". The mount is already rwxrwxrwx, so agent can still
    # read/write snapshot tarballs on it.
    print("  Creating agent user + local state dir...")
    sandbox.process.exec(
        f"useradd -m -s /bin/bash agent && "
        f"mkdir -p {LOCAL_STATE}/data {LOCAL_STATE}/system {LOCAL_STATE}/cache && "
        f"chown -R agent:agent /workspace /opt/cognee-integrations {LOCAL_STATE} && "
        "env | grep -E '^(ANTHROPIC_API_KEY|LLM_|COGNEE_|CACHING|VECTOR_|"
        "DATA_ROOT_DIRECTORY|SYSTEM_ROOT_DIRECTORY|CACHE_ROOT_DIRECTORY)=' "
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
    """Run one Claude Code task with sync-in / run / sync-out around it.

    The cognee DBs live on local sandbox disk (block storage). Vectors go
    to Moss (cloud) and the vector search happens in-process. Before the
    agent runs we extract the latest shared snapshot from the volume into
    LOCAL_STATE; after the agent
    finishes we tar LOCAL_STATE back into the volume as a single object.
    Whole-object writes are what mountpoint-s3 is built for.
    """
    print(f"\n{'=' * 60}")
    print(f"  AGENT:  {label}")
    print(f"  PROMPT: {prompt[:80]}...")
    print(f"{'=' * 60}\n")

    plugin_dir = f"/opt/cognee-integrations/{PLUGIN_SUBPATH}"
    snapshot_path = f"{MOUNT_PATH}/{SNAPSHOT_FILE}"

    # One bash block: sync-in, run claude, sync-out. The sync-out is in a
    # trap so the snapshot gets written even if claude errors.
    inner = (
        f"set -a && . /home/agent/env && set +a && "
        f"sync_out() {{ "
        f"  tar -czf /tmp/new-state.tar.gz -C {LOCAL_STATE} . && "
        f"  cp /tmp/new-state.tar.gz {shlex.quote(snapshot_path)} && "
        f"  echo '[sync-out] wrote '$(stat -c%s /tmp/new-state.tar.gz)' bytes'; "
        f"}}; trap sync_out EXIT; "
        f"if [ -f {shlex.quote(snapshot_path)} ]; then "
        f"  echo '[sync-in] found prior snapshot ('$(stat -c%s {shlex.quote(snapshot_path)})' bytes), extracting'; "
        f"  tar -xzf {shlex.quote(snapshot_path)} -C {LOCAL_STATE}; "
        f"else "
        f"  echo '[sync-in] no prior snapshot, starting fresh'; "
        f"fi; "
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
# Graph inspection (post-run verification)
# ---------------------------------------------------------------------------


def inspect_shared_graph(daytona, volume_id):
    """Mount the volume, extract the snapshot to local disk, and report
    row counts from the SQLite DB. This is the proof that the shared
    graph actually carries data across agents."""
    print("\n=== Inspecting shared graph ===")
    sandbox = _create_with_quota_retry(
        daytona,
        CreateSandboxFromImageParams(
            image=Image.debian_slim("3.12"),
            resources=Resources(cpu=1, memory=2, disk=5),
            volumes=[VolumeMount(volume_id=volume_id, mount_path=MOUNT_PATH)],
            labels={"app": "cognee-inspector", "demo": "shared-memory"},
        ),
    )
    try:
        snapshot_path = f"{MOUNT_PATH}/{SNAPSHOT_FILE}"
        # Extract snapshot, then SQLite query the cognee_db.
        query = (
            "import sqlite3, glob\n"
            f"db_paths = glob.glob('{LOCAL_STATE}/system/databases/cognee_db') "
            f"or glob.glob('{LOCAL_STATE}/system/**/cognee_db', recursive=True)\n"
            "if not db_paths:\n"
            "    print('no sqlite found in snapshot'); raise SystemExit\n"
            "c = sqlite3.connect(db_paths[0]).cursor()\n"
            "for t in ('datasets', 'data', 'pipeline_runs', 'users', 'nodes', 'edges'):\n"
            "    try:\n"
            "        n = c.execute(f'select count(*) from \"{t}\"').fetchone()[0]\n"
            "        print(f'{t}: {n}')\n"
            "    except Exception as e:\n"
            "        print(f'{t}: err {e}')\n"
        )
        import base64

        b64 = base64.b64encode(query.encode()).decode()
        extract_and_query = (
            f"mkdir -p {LOCAL_STATE} && "
            f"if [ -f {shlex.quote(snapshot_path)} ]; then "
            f"  echo 'snapshot size: '$(stat -c%s {shlex.quote(snapshot_path)})' bytes'; "
            f"  tar -xzf {shlex.quote(snapshot_path)} -C {LOCAL_STATE}; "
            f"  echo '---contents---'; find {LOCAL_STATE} -type f | head -20; "
            f"  echo '---sqlite counts---'; "
            f"  echo {b64} | base64 -d > /tmp/q.py && python /tmp/q.py; "
            f"else "
            f"  echo 'NO_SNAPSHOT at {snapshot_path}'; "
            f"fi"
        )
        r = sandbox.process.exec(extract_and_query, timeout=60)
        print(r.result or "(no output)")
    finally:
        daytona.delete(sandbox)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code + Cognee + Moss on Daytona: shared-volume memory demo"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("TARGET_REPO", DEFAULT_REPO),
        help="Git URL of the repository the agents will explore",
    )
    parser.add_argument(
        "--keep-volume",
        action="store_true",
        help="Don't delete the shared volume at the end (useful for inspection)",
    )
    parser.add_argument(
        "--cognee-install-spec",
        default=os.environ.get("COGNEE_INSTALL_SPEC", DEFAULT_COGNEE_INSTALL_SPEC),
        help=(
            "Cognee package, wheel path, or git URL to install in each sandbox "
            f"(default: {DEFAULT_COGNEE_INSTALL_SPEC})"
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CLAUDE CODE + COGNEE SHARED MEMORY DEMO (volume)")
    print(f"  Target repo: {args.repo}")
    print(f"  Cognee install: {args.cognee_install_spec}")
    print("=" * 60 + "\n")

    daytona = _get_daytona()
    volume = ensure_shared_volume(daytona)

    # Three agents with increasingly broad prompts. Each agent's session
    # cache is private (per-directory, scoped by COGNEE_SESSION_PREFIX);
    # anything they /cognee-memory:cognee-remember or SessionEnd-bridge
    # lands in the shared graph on the volume.
    agents = [
        (
            "arch",
            "Explore this codebase. Map the project structure, key entry "
            "points, main modules, and overall architecture. Use "
            "/cognee-memory:cognee-remember to store the most important "
            "findings so other agents can read them.",
        ),
        (
            "api",
            "Before exploring, use /cognee-memory:cognee-search to recall "
            "anything the architecture agent stored. Then describe the API "
            "layer — routes, middleware, auth, request handling. Store "
            "your findings with /cognee-memory:cognee-remember.",
        ),
        (
            "tests",
            "Use /cognee-memory:cognee-search to pull in what the "
            "architecture and API agents already learned. Then describe "
            "the testing strategy — which suites exist, what they cover, "
            "and any gaps. Store findings with /cognee-memory:cognee-remember.",
        ),
    ]

    ran = []
    try:
        for label, prompt in agents:
            sandbox = deploy_claude_code_agent(
                volume.id,
                args.repo,
                label,
                args.cognee_install_spec,
            )
            try:
                run_agent_task(sandbox, label, prompt)
                ran.append((label, sandbox.id))

                # Dump the plugin's resolved state so we can see whether
                # SessionStart completed and what mode it picked.
                try:
                    dump = sandbox.process.exec(
                        "cat /home/agent/.cognee-plugin/resolved.json 2>&1 || "
                        "echo MISSING_RESOLVED_JSON",
                        timeout=5,
                    )
                    if dump.result:
                        print(f"  [{label}] resolved.json:\n{dump.result}")
                except Exception:
                    pass
            finally:
                try:
                    daytona.delete(sandbox)
                    print(f"  [{label}] sandbox {sandbox.id} destroyed.")
                except Exception as e:
                    print(f"  [{label}] WARNING: failed to delete sandbox: {e}")

        inspect_shared_graph(daytona, volume.id)
    finally:
        if not args.keep_volume:
            try:
                daytona.volume.delete(volume)
                print(f"\n  Shared volume {volume.name} deleted.")
            except Exception as e:
                print(f"\n  WARNING: failed to delete volume: {e}")
        else:
            print(f"\n  Shared volume {volume.name} ({volume.id}) kept.")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    for label, sandbox_id in ran:
        print(f"  Agent {label:<6} ran in: {sandbox_id} (destroyed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
