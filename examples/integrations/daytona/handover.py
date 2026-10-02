"""Multi-agent memory handover on Daytona sandboxes, as a fork chain.

The Daytona port of ``examples/integrations/docker-sandbox-kit/demo/handover.sh``.
The payload is the same ``supervisor_worker_handover.py``: two cognee users
(supervisor, worker) sharing one embedded cognee deployment, protected by
cognee's ACLs. What changes is how the memory travels between sandboxes.

Docker Sandboxes hand the cognee state around with ``sbx cp``. Daytona can
fork a running sandbox into a copy-on-write clone of its filesystem, so the
handover *is* the fork::

    snapshot "cognee-memory"  (debian + cognee, built once, reused)
      -> cognee-supervisor      --phase brief    (state on sandbox-local disk)
      -> fork: cognee-worker    --phase work     (recall by UUID, prove denials, report)
      -> fork: cognee-review    --phase review   (supervisor reads the report)

Every sandbox in the chain runs with:

- ``LLM_API_KEY`` bound to an organization Secret scoped to ``api.openai.com``.
  The sandbox sees an opaque placeholder; Daytona's proxy substitutes the real
  key on requests to that host. The key never enters any sandbox.
- a domain allow list that holds only what cognee needs at runtime. PyPI is not
  on it: cognee is baked into the snapshot.

Prerequisites::

    pip install daytona
    export DAYTONA_API_KEY=...        # https://app.daytona.io
    export LLM_API_KEY=sk-...         # only needed the first time, to create the Secret

Usage::

    python handover.py                # full round trip, sandboxes deleted afterwards
    python handover.py --keep         # leave the three sandboxes for inspection
    python handover.py --rebuild      # rebuild the snapshot (e.g. after a cognee release)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from daytona import (  # type: ignore[import-untyped]
    CreateSandboxFromSnapshotParams,
    CreateSecretParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaError,
    Image,
    Resources,
    Sandbox,
)

SNAPSHOT_NAME = os.environ.get("COGNEE_SNAPSHOT", "cognee-memory")
SECRET_NAME = os.environ.get("COGNEE_SECRET", "cognee-openai")
LLM_HOST = "api.openai.com"
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-5.6-luna")

# Runtime egress. Derived from the Docker Sandboxes kit, which was run under a
# deny-all policy and read back from the policy log; minus PyPI, because the
# install happens at snapshot build time, not inside the sandbox.
RUNTIME_DOMAINS = (
    LLM_HOST,
    "extension.ladybugdb.com",  # embedded graph DB fetches extensions on first use
    "raw.githubusercontent.com",  # litellm's model-cost map
)
DOMAIN_ALLOW_LIST = ",".join(RUNTIME_DOMAINS)

# Paths inside the sandbox, relative to the sandbox user's home.
STATE_DIR = "cognee-state"
DEMO_DIR = "handover-demo"
PAYLOAD_NAME = "supervisor_worker_handover.py"
TOKEN_REL_PATH = f"{DEMO_DIR}/handover-out/handover_token.json"

# Single source of truth for the payload: the Docker Sandboxes demo next door.
DEFAULT_PAYLOAD = (
    Path(__file__).resolve().parent.parent / "docker-sandbox-kit" / "demo" / PAYLOAD_NAME
)

PHASES = (
    ("cognee-supervisor", "brief"),
    ("cognee-worker", "work"),
    ("cognee-review", "review"),
)


def ensure_secret(daytona: Daytona) -> str:
    """Return the placeholder of the host-scoped LLM secret, creating it on first run."""
    existing = [s for s in daytona.secret.list(name=SECRET_NAME).items if s.name == SECRET_NAME]
    if existing:
        return existing[0].placeholder

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise SystemExit(
            f"Secret '{SECRET_NAME}' does not exist yet and LLM_API_KEY is not set. "
            "Export LLM_API_KEY once so the Secret can be created; later runs do not need it."
        )
    print(f"=== creating Secret '{SECRET_NAME}' (scoped to {LLM_HOST}) ===")
    created = daytona.secret.create(
        CreateSecretParams(
            name=SECRET_NAME,
            value=llm_api_key,
            hosts=[LLM_HOST],
            description="OpenAI key used by cognee for entity extraction and embeddings",
        )
    )
    return created.placeholder


def ensure_snapshot(daytona: Daytona, rebuild: bool) -> None:
    """Build the cognee snapshot once; every sandbox in the chain starts from it."""
    if not rebuild:
        try:
            daytona.snapshot.get(SNAPSHOT_NAME)
            print(f"=== snapshot '{SNAPSHOT_NAME}' exists, reusing ===")
            return
        except DaytonaError:
            pass
    else:
        try:
            daytona.snapshot.delete(daytona.snapshot.get(SNAPSHOT_NAME))
        except DaytonaError:
            pass

    print(f"=== building snapshot '{SNAPSHOT_NAME}' (installs cognee; several minutes) ===")
    image = Image.debian_slim("3.12").pip_install("cognee")
    daytona.snapshot.create(
        CreateSnapshotParams(
            name=SNAPSHOT_NAME,
            image=image,
            resources=Resources(cpu=2, memory=4, disk=10),
        ),
        on_logs=lambda line: print(f"[snapshot] {line}", end="", flush=True),
    )
    print()


def run_in(sandbox: Sandbox, command: str, *, cwd: str | None = None, env=None, timeout=1800):
    """Run a shell command in the sandbox, echo its output, fail loudly on non-zero exit."""
    response = sandbox.process.exec(command, cwd=cwd, env=env, timeout=timeout)
    output = response.result or ""
    for line in output.splitlines():
        print(f"    {line}")
    if response.exit_code not in (0, None):
        raise RuntimeError(f"command failed with exit {response.exit_code}: {command}")
    return output


def cognee_env(home: str) -> dict[str, str]:
    """The cognee configuration every phase runs with (mirrors the sbx kit's environment block)."""
    return {
        "DATA_ROOT_DIRECTORY": f"{home}/{STATE_DIR}/data",
        "SYSTEM_ROOT_DIRECTORY": f"{home}/{STATE_DIR}/system",
        "LLM_MODEL": LLM_MODEL,
        "TELEMETRY_DISABLED": "1",
        "LOG_LEVEL": "ERROR",
        # Multi-tenant ACLs + per-user+dataset DB isolation. cognee's default,
        # pinned so the demo is explicit about the boundary it relies on.
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
    }


def create_supervisor(daytona: Daytona, payload: Path) -> Sandbox:
    """First sandbox in the chain: fresh from the snapshot, payload uploaded, empty memory."""
    print("=== creating sandbox: cognee-supervisor (from snapshot) ===")
    sandbox = daytona.create(
        CreateSandboxFromSnapshotParams(
            snapshot=SNAPSHOT_NAME,
            name="cognee-supervisor",
            # Env var -> Secret name. The sandbox sees the placeholder only.
            secrets={"LLM_API_KEY": SECRET_NAME},
            domain_allow_list=DOMAIN_ALLOW_LIST,
            labels={"app": "cognee", "demo": "handover", "role": "supervisor"},
            auto_stop_interval=30,
        ),
        timeout=180,
    )
    home = sandbox.get_user_home_dir()
    run_in(sandbox, f"mkdir -p {home}/{STATE_DIR} {home}/{DEMO_DIR}/handover-out")
    sandbox.fs.upload_file(payload.read_bytes(), f"{home}/{DEMO_DIR}/{PAYLOAD_NAME}")
    return sandbox


def fork_into(parent: Sandbox, name: str, role: str) -> Sandbox:
    """Hand the memory over: a copy-on-write clone of the parent's disk, same policy."""
    print(f"=== forking {parent.name} -> {name} (the memory handover) ===")
    child = parent.fork(name=name, timeout=180)
    # Fork copies the disk; re-assert the security posture on the new sandbox
    # rather than assume it is inherited.
    child.update_secrets({"LLM_API_KEY": SECRET_NAME})
    child.update_network_settings(domain_allow_list=DOMAIN_ALLOW_LIST)
    child.set_labels({"app": "cognee", "demo": "handover", "role": role})
    return child


def run_phase(sandbox: Sandbox, phase: str) -> None:
    print(f"\n=== sandbox: {sandbox.name} (phase: {phase}) ===")
    home = sandbox.get_user_home_dir()
    run_in(
        sandbox,
        f"python3 {PAYLOAD_NAME} --phase {phase} --token-file handover-out/handover_token.json",
        cwd=f"{home}/{DEMO_DIR}",
        env=cognee_env(home),
    )


def show_boundaries(sandbox: Sandbox, placeholder: str) -> None:
    """Prove the two isolation properties the kit promises, from inside the sandbox."""
    print(f"\n=== security check inside {sandbox.name} ===")
    seen = run_in(sandbox, "printf '%s' \"$LLM_API_KEY\"")
    print(
        f"    LLM_API_KEY inside sandbox is {'the placeholder' if seen == placeholder else seen!r}"
    )
    probe = sandbox.process.exec(
        "python3 -c \"import urllib.request;urllib.request.urlopen('https://example.com',timeout=5)\"",
        timeout=20,
    )
    verdict = (
        "blocked (not on the allow list)"
        if probe.exit_code
        else "REACHABLE - allow list not applied!"
    )
    print(f"    egress to example.com: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--payload", type=Path, default=DEFAULT_PAYLOAD, help="path to the handover payload"
    )
    parser.add_argument(
        "--keep", action="store_true", help="do not delete the sandboxes afterwards"
    )
    parser.add_argument("--rebuild", action="store_true", help="rebuild the cognee snapshot")
    args = parser.parse_args()

    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is required (https://app.daytona.io)")
    if not args.payload.is_file():
        raise SystemExit(f"payload not found: {args.payload}")

    daytona = Daytona()  # reads DAYTONA_API_KEY / DAYTONA_API_URL from the environment
    placeholder = ensure_secret(daytona)
    ensure_snapshot(daytona, rebuild=args.rebuild)

    sandboxes: list[Sandbox] = []
    try:
        supervisor = create_supervisor(daytona, args.payload)
        sandboxes.append(supervisor)
        run_phase(supervisor, "brief")
        show_boundaries(supervisor, placeholder)

        worker = fork_into(supervisor, "cognee-worker", role="worker")
        sandboxes.append(worker)
        run_phase(worker, "work")

        review = fork_into(worker, "cognee-review", role="supervisor-review")
        sandboxes.append(review)
        run_phase(review, "review")

        token = review.fs.download_file(f"{review.get_user_home_dir()}/{TOKEN_REL_PATH}")
        out = Path(__file__).resolve().parent / "handover-out"
        out.mkdir(exist_ok=True)
        (out / "handover_token.json").write_bytes(token or b"")

        print("\nHandover round trip passed across three forked Daytona sandboxes.")
        print(f"Token exchanged via: {out / 'handover_token.json'}")
        print(f"Dataset shared by UUID: {json.loads(token or b'{}').get('dataset_id')}")
    finally:
        if args.keep:
            print("\nSandboxes kept: " + ", ".join(s.name for s in sandboxes))
            print(
                'Delete them with: python -c "from daytona import Daytona; d=Daytona(); '
                "[d.delete(d.get(n)) for n in ('cognee-supervisor','cognee-worker','cognee-review')]\""
            )
        else:
            for sandbox in sandboxes:
                try:
                    daytona.delete(sandbox)
                except DaytonaError as err:
                    print(f"could not delete {sandbox.name}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
