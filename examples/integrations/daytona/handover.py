"""Multi-agent memory handover on Daytona sandboxes, as a fork chain.

The Daytona port of ``examples/integrations/docker-sandbox-kit/demo/handover.sh``.
The payload is the same ``supervisor_worker_handover.py``: two cognee users
(supervisor, worker) sharing one embedded cognee deployment, protected by
cognee's ACLs. What changes is how the memory travels between sandboxes.

Docker Sandboxes hand the cognee state around with ``sbx cp``. Daytona can
fork a running sandbox into a copy-on-write clone of its filesystem, so the
handover *is* the fork::

    snapshot "cognee-memory"  (debian + cognee, built once, reused)
      -> cognee-supervisor-<run>      --phase brief    (state on sandbox-local disk)
      -> fork: cognee-worker-<run>    --phase work     (recall by UUID, prove denials, report)
      -> fork: cognee-review-<run>    --phase review   (supervisor reads the report)

``<run>`` is a random suffix minted per invocation, so a run never collides
with sandboxes a previous ``--keep`` left behind, and every sandbox carries a
``run=<run>`` label the cleanup sweeps by.

Every sandbox in the chain runs with:

- ``LLM_API_KEY`` bound to an organization Secret scoped to ``api.openai.com``.
  The sandbox sees an opaque placeholder; Daytona's proxy substitutes the real
  key on requests to that host. The key never enters any sandbox.
- a domain allow list that holds only what cognee needs at runtime. PyPI is not
  on it: cognee is baked into the snapshot. Neither is GitHub: litellm is told
  to use its bundled model-cost map instead of fetching it.

The walls are checked, not just shown: after the first phase the script proves
from inside the sandbox that ``$LLM_API_KEY`` is the placeholder and that a
host off the allow list is unreachable on both the explicit-proxy path and the
direct path (a client that ignores ``HTTP(S)_PROXY`` must hit the same wall).
A wall that does not hold fails the run; sandboxes are still cleaned up.

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
import secrets
import sys
from pathlib import Path

from daytona import (  # type: ignore[import-untyped]
    CreateSandboxFromSnapshotParams,
    CreateSecretParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaError,
    Image,
    ListSandboxesQuery,
    Resources,
    Sandbox,
)

SNAPSHOT_NAME = os.environ.get("COGNEE_SNAPSHOT", "cognee-memory")
# Name of the Daytona Secret that holds the LLM key. Only this label is ever
# printed or passed around; the key value itself never leaves Daytona's vault.
LLM_VAULT_ENTRY = os.environ.get("COGNEE_SECRET", "cognee-openai")
LLM_HOST = "api.openai.com"
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-5.6-luna")

# Runtime egress. Derived from the Docker Sandboxes kit, which was run under a
# deny-all policy and read back from the policy log; minus PyPI, because the
# install happens at snapshot build time, not inside the sandbox; minus
# raw.githubusercontent.com, because LITELLM_LOCAL_MODEL_COST_MAP (below) makes
# litellm use the model-cost map bundled in the package instead of fetching it.
# A domain on this list is open for the sandbox's whole life, so every entry
# is one more place an agent can read from; keep it at what cognee cannot run
# without.
RUNTIME_DOMAINS = (
    LLM_HOST,
    # The Linux ladybug wheel downloads its JSON extension on first use. cognee
    # ships bundled binaries (cognee_db_workers/ladybug_extensions/) and only
    # falls back to the download when none matches; the host stays listed so
    # a wheel without a matching bundle still works.
    "extension.ladybugdb.com",
)
DOMAIN_ALLOW_LIST = ",".join(RUNTIME_DOMAINS)

# A host that is NOT on the allow list; the egress probe must fail to reach it.
EGRESS_PROBE_URL = "https://example.com"

# Minutes of inactivity before a sandbox stops. Set on the first sandbox and
# re-asserted on every fork, like the rest of the posture.
AUTO_STOP_MINUTES = 30

# Paths inside the sandbox, relative to the sandbox user's home.
STATE_DIR = "cognee-state"
DEMO_DIR = "handover-demo"
PAYLOAD_NAME = "supervisor_worker_handover.py"
TOKEN_REL_PATH = f"{DEMO_DIR}/handover-out/handover_token.json"

# Single source of truth for the payload: the Docker Sandboxes demo next door.
DEFAULT_PAYLOAD = (
    Path(__file__).resolve().parent.parent / "docker-sandbox-kit" / "demo" / PAYLOAD_NAME
)


def run_labels(run_id: str, role: str) -> dict[str, str]:
    """Labels every sandbox of one run carries; cleanup lists by `run`."""
    return {"app": "cognee", "demo": "handover", "run": run_id, "role": role}


def ensure_secret(daytona: Daytona) -> str:
    """Return the placeholder of the host-scoped LLM secret, creating it on first run."""
    existing = [
        s for s in daytona.secret.list(name=LLM_VAULT_ENTRY).items if s.name == LLM_VAULT_ENTRY
    ]
    if existing:
        return existing[0].placeholder

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        raise SystemExit(
            f"Secret '{LLM_VAULT_ENTRY}' does not exist yet and LLM_API_KEY is not set. "
            "Export LLM_API_KEY once so the Secret can be created; later runs do not need it."
        )
    print(f"=== creating Secret '{LLM_VAULT_ENTRY}' (scoped to {LLM_HOST}) ===")
    created = daytona.secret.create(
        CreateSecretParams(
            name=LLM_VAULT_ENTRY,
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
        # Use the model-cost map bundled with litellm instead of fetching it
        # from raw.githubusercontent.com, so that host can stay off the allow list.
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }


def create_supervisor(daytona: Daytona, payload: Path, run_id: str) -> Sandbox:
    """First sandbox in the chain: fresh from the snapshot, payload uploaded, empty memory."""
    name = f"cognee-supervisor-{run_id}"
    print(f"=== creating sandbox: {name} (from snapshot) ===")
    sandbox = daytona.create(
        CreateSandboxFromSnapshotParams(
            snapshot=SNAPSHOT_NAME,
            name=name,
            # Env var -> Secret name. The sandbox sees the placeholder only.
            secrets={"LLM_API_KEY": LLM_VAULT_ENTRY},
            domain_allow_list=DOMAIN_ALLOW_LIST,
            labels=run_labels(run_id, "supervisor"),
            auto_stop_interval=AUTO_STOP_MINUTES,
        ),
        timeout=180,
    )
    home = sandbox.get_user_home_dir()
    run_in(sandbox, f"mkdir -p {home}/{STATE_DIR} {home}/{DEMO_DIR}/handover-out")
    sandbox.fs.upload_file(payload.read_bytes(), f"{home}/{DEMO_DIR}/{PAYLOAD_NAME}")
    return sandbox


def fork_into(
    parent: Sandbox, name: str, role: str, run_id: str, created: list[Sandbox]
) -> Sandbox:
    """Hand the memory over: a copy-on-write clone of the parent's disk, same policy.

    The child is appended to `created` the moment it exists, before its posture
    is re-asserted: if `update_secrets` or `update_network_settings` fails, the
    run aborts and cleanup must still find the sandbox.
    """
    print(f"=== forking {parent.name} -> {name} (the memory handover) ===")
    child = parent.fork(name=name, timeout=180)
    created.append(child)
    # Fork copies the disk; re-assert the security posture on the new sandbox
    # rather than assume it is inherited.
    child.update_secrets({"LLM_API_KEY": LLM_VAULT_ENTRY})
    child.update_network_settings(domain_allow_list=DOMAIN_ALLOW_LIST)
    child.set_autostop_interval(AUTO_STOP_MINUTES)
    child.set_labels(run_labels(run_id, role))
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


# Two ways out of the sandbox. Daytona enforces the domain allow list and
# substitutes Secrets in its egress proxy, which clients reach through the
# HTTP(S)_PROXY variables it sets. A client that ignores those variables (the
# way aiohttp did before cognee #5196) takes the direct path, so the wall has
# to hold there too; the second probe disables proxy handling to take it.
_PROXY_PROBE = f"import urllib.request as u;u.urlopen('{EGRESS_PROBE_URL}',timeout=5)"
_DIRECT_PROBE = (
    f"import urllib.request as u;"
    f"u.build_opener(u.ProxyHandler({{}})).open('{EGRESS_PROBE_URL}',timeout=5)"
)
EGRESS_PROBES = (
    ("proxy path", f'python3 -c "{_PROXY_PROBE}"'),
    ("direct path (proxy env ignored)", f'python3 -c "{_DIRECT_PROBE}"'),
)


def check_boundaries(sandbox: Sandbox, placeholder: str) -> None:
    """Prove the isolation properties from inside the sandbox; fail the run if one is missing.

    A wall that does not hold must not end in "round trip passed": every
    verdict is printed, then the first breach raises and the caller's cleanup
    still runs.
    """
    print(f"\n=== security check inside {sandbox.name} ===")
    breaches: list[str] = []

    # Read the value without `run_in` (which echoes command output): if the
    # Secret binding failed, `seen` would be the real key. Report the verdict
    # and the length only, never the value.
    seen = (sandbox.process.exec("printf '%s' \"$LLM_API_KEY\"", timeout=20).result or "").strip()
    if seen == placeholder:
        print("    LLM_API_KEY inside sandbox is the placeholder")
    else:
        verdict = f"LLM_API_KEY is NOT the placeholder ({len(seen)} chars): Secret binding failed"
        print(f"    {verdict}")
        breaches.append(verdict)

    for label, command in EGRESS_PROBES:
        probe = sandbox.process.exec(command, timeout=20)
        if probe.exit_code:
            print(f"    egress to {EGRESS_PROBE_URL} via {label}: blocked (not on the allow list)")
        else:
            verdict = (
                f"egress to {EGRESS_PROBE_URL} via {label}: REACHABLE, allow list not enforced"
            )
            print(f"    {verdict}")
            breaches.append(verdict)

    if breaches:
        raise RuntimeError(f"isolation breached in {sandbox.name}: " + "; ".join(breaches))


def sweep_run(daytona: Daytona, run_id: str, created: list[Sandbox]) -> None:
    """Add every sandbox labelled with this run that the script lost track of.

    `daytona.create` can time out after the sandbox exists, and a fork can fail
    after the clone is up; both leave a running sandbox nothing holds a handle
    to. The labels are set at creation, so listing by `run` finds them.
    """
    known = {sandbox.id for sandbox in created}
    try:
        for sandbox in daytona.list(ListSandboxesQuery(labels={"demo": "handover", "run": run_id})):
            if sandbox.id not in known:
                created.append(sandbox)
    except DaytonaError as err:
        print(f"could not list sandboxes of run {run_id}: {err}", file=sys.stderr)


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

    # One id per invocation: sandbox names carry it (no collision with a
    # previous --keep run) and so does a label (cleanup lists by it).
    run_id = secrets.token_hex(3)
    print(f"=== run {run_id} ===")

    sandboxes: list[Sandbox] = []
    leaked: list[str] = []
    try:
        supervisor = create_supervisor(daytona, args.payload, run_id)
        sandboxes.append(supervisor)
        run_phase(supervisor, "brief")
        check_boundaries(supervisor, placeholder)

        worker = fork_into(supervisor, f"cognee-worker-{run_id}", "worker", run_id, sandboxes)
        run_phase(worker, "work")

        review = fork_into(
            worker, f"cognee-review-{run_id}", "supervisor-review", run_id, sandboxes
        )
        run_phase(review, "review")

        token = review.fs.download_file(f"{review.get_user_home_dir()}/{TOKEN_REL_PATH}")
        out = Path(__file__).resolve().parent / "handover-out"
        out.mkdir(exist_ok=True)
        (out / "handover_token.json").write_bytes(token or b"")

        print("\nHandover round trip passed across three forked Daytona sandboxes.")
        print(f"Token exchanged via: {out / 'handover_token.json'}")
        print(f"Dataset shared by UUID: {json.loads(token or b'{}').get('dataset_id')}")
    finally:
        sweep_run(daytona, run_id, sandboxes)
        if args.keep:
            print("\nSandboxes kept: " + ", ".join(s.name for s in sandboxes))
            print(
                'Delete them with: python -c "from daytona import Daytona, ListSandboxesQuery; '
                "d=Daytona(); [d.delete(s) for s in d.list(ListSandboxesQuery("
                f"labels={{'demo': 'handover', 'run': '{run_id}'}}))]\""
            )
        else:
            for sandbox in sandboxes:
                try:
                    daytona.delete(sandbox)
                except DaytonaError as err:
                    leaked.append(sandbox.name)
                    print(f"could not delete {sandbox.name}: {err}", file=sys.stderr)
    if leaked:
        # A sandbox left running keeps a mounted Secret and an open allow
        # list; that is a failed cleanup, not a passed run. (Raised outside the
        # finally block so it never hides the traceback of an earlier failure.)
        raise SystemExit(f"sandboxes still running: {', '.join(leaked)}")


if __name__ == "__main__":
    main()
