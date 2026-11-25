"""Multi-agent memory handover on Daytona sandboxes, one sandbox per phase.

The Daytona port of ``examples/integrations/docker-sandbox-kit/demo/handover.sh``.
The payload is the same ``supervisor_worker_handover.py``: two cognee users
(supervisor, worker) sharing one embedded cognee deployment, protected by
cognee's ACLs. What changes is how the memory travels between sandboxes::

    snapshot "cognee-memory"  (debian + cognee, built once, reused)
      -> cognee-supervisor-<run>          --phase brief    (state on sandbox-local disk)
      -> handover: cognee-worker-<run>    --phase work     (recall by UUID, prove denials, report)
      -> handover: cognee-review-<run>    --phase review   (supervisor reads the report)

A handover moves the shared dataset, by default as a **COGX** archive —
cognee's exchange format: ``cognee.export(format="cogx",
include_permissions=True)`` in the parent, ``cognee.remember(COGXArchiveSource)``
in a fresh sandbox from the snapshot (see ``cogx_handover.py``). Only the
shared dataset travels, with its owner and ACL grants; the supervisor's private
dataset never leaves the first sandbox. Two raw-state alternatives remain for
comparison (``--handover copy`` / ``fork``): the whole ``~/cognee-state``
directory archived into a fresh sandbox (the ``sbx cp`` mechanism of the Docker
demo), or a copy-on-write fork of the running VM (Linux VM sandboxes only;
container sandboxes answer ``422 Forking is not supported``, and ``auto`` falls
back to the copy). A sandbox is stopped as soon as its phase is done and its
state handed over.

``<run>`` is a random suffix minted per invocation, so a run never collides
with sandboxes a previous ``--keep`` left behind, and every sandbox carries a
``run=<run>`` label the cleanup sweeps by.

Every sandbox in the chain runs with:

- ``LLM_API_KEY`` bound to an organization Secret scoped to ``api.openai.com``.
  The sandbox sees an opaque placeholder; Daytona's proxy substitutes the real
  key on requests to that host. The key never enters any sandbox.
- a domain allow list that holds only what cognee needs at runtime (Tier 3/4
  organizations; on Tier 1/2 Daytona rejects a sandbox-level list and the
  organization's tier policy applies instead — the run says which it got).
  PyPI is not on it: cognee is baked into the snapshot. Neither is GitHub:
  litellm is told to use its bundled model-cost map instead of fetching it.

The walls are checked, not just shown: after the first phase the script proves
from inside the sandbox that ``$LLM_API_KEY`` is the placeholder, that the
placeholder does not authenticate outside the proxy, and that a host off the
allow list is unreachable on both the explicit-proxy path and the direct path
(a client that ignores ``HTTP(S)_PROXY`` must hit the same wall). A wall that
does not hold fails the run; sandboxes are still cleaned up. One documented
exception: on Tier 1/2 the tier policy is enforced on direct connections only
and Daytona's egress proxy forwards anywhere; that is reported as a WARNING
(``--strict-walls`` makes it a failure).

Prerequisites::

    pip install daytona
    export DAYTONA_API_KEY=...        # https://app.daytona.io
    export LLM_API_KEY=sk-...         # only needed the first time, to create the Secret

Usage::

    python handover.py                # full round trip, sandboxes deleted afterwards
    python handover.py --keep         # leave the three sandboxes for inspection
    python handover.py --rebuild      # rebuild the snapshot (e.g. after a cognee release)
    python handover.py --handover copy    # raw state copy instead of a COGX archive
    python handover.py --strict-walls     # fail on the Tier 1/2 proxy-path gap
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

from daytona import (  # type: ignore[import-untyped]
    CreateSandboxFromSnapshotParams,
    CreateSecretParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaBadRequestError,
    DaytonaError,
    Image,
    ListSandboxesQuery,
    Resources,
    Sandbox,
)
from daytona_api_client import SnapshotState  # type: ignore[import-untyped]

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

# Which network wall a run ends up behind. Daytona applies the sandbox-level
# allow list above only on Tier 3/4 organizations. On Tier 1/2 the policy is
# set at the organization level and a sandbox-level list is rejected (HTTP
# 400 with the message below): the sandbox then sits behind Daytona's default
# tier policy, which blocks everything except its "essential services"
# (package registries, git hosts, model providers — api.openai.com among
# them). That is a wider wall than the two hosts above, so a run says which
# one it got. https://www.daytona.io/docs/en/network-limits/
NETWORK_POLICY_SANDBOX = "sandbox allow list"
NETWORK_POLICY_ORGANIZATION = "organization tier policy (essential services only)"
_ORG_POLICY_LOCKED = "cannot be overridden at the sandbox level"

# A host that is on neither wall; the egress probe must fail to reach it.
EGRESS_PROBE_URL = "https://example.com"

# Minutes of inactivity before a sandbox stops. Set on the first sandbox and
# re-asserted on every fork, like the rest of the posture.
AUTO_STOP_MINUTES = 30

# Paths inside the sandbox, relative to the sandbox user's home.
STATE_DIR = "cognee-state"
DEMO_DIR = "handover-demo"
PAYLOAD_NAME = "supervisor_worker_handover.py"
HELPER_NAME = "cogx_handover.py"  # export/import of the shared dataset, runs in the sandbox
TOKEN_REL_PATH = f"{DEMO_DIR}/handover-out/handover_token.json"
# The dataset owner the COGX export runs as (mirrors the payload's constant).
SUPERVISOR_EMAIL = "supervisor@handover.demo"

# Single source of truth for the payload: the Docker Sandboxes demo next door.
DEFAULT_PAYLOAD = (
    Path(__file__).resolve().parent.parent / "docker-sandbox-kit" / "demo" / PAYLOAD_NAME
)
HELPER = Path(__file__).resolve().parent / HELPER_NAME


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
    """Build the cognee snapshot once; every sandbox in the chain starts from it.

    A snapshot that sits unused for a while is deactivated by Daytona (state
    ``inactive``) and a ``create`` from it is refused with ``400 Snapshot ... is
    inactive``. Existing is not enough: an inactive snapshot is activated and
    waited for before the first sandbox is created.
    """
    if not rebuild:
        try:
            snapshot = daytona.snapshot.get(SNAPSHOT_NAME)
        except DaytonaError:
            snapshot = None
        if snapshot is not None:
            if snapshot.state == SnapshotState.ACTIVE:
                print(f"=== snapshot '{SNAPSHOT_NAME}' exists, reusing ===")
            else:
                print(
                    f"=== snapshot '{SNAPSHOT_NAME}' exists but is {snapshot.state.value}: activating ==="
                )
                activate_snapshot(daytona)
            return
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


def activate_snapshot(daytona: Daytona, timeout: float = 900, name: str = SNAPSHOT_NAME) -> None:
    """Bring an inactive snapshot back to ``active`` and wait until it is.

    Activating a declaratively built snapshot makes Daytona rebuild it (the
    state goes through ``building``), which takes about as long as the first
    build did. A snapshot already on its way (``building``, ``pending``,
    ``pulling``, e.g. from an interrupted earlier run) is only waited for.
    """
    if daytona.snapshot.get(name).state == SnapshotState.INACTIVE:
        daytona.snapshot.activate(name)
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        state = daytona.snapshot.get(name).state
        if state != last:
            print(f"    snapshot state: {state.value}")
            last = state
        if state == SnapshotState.ACTIVE:
            return
        if state in (SnapshotState.ERROR, SnapshotState.BUILD_FAILED):
            raise RuntimeError(f"snapshot '{name}' failed to activate: {state.value}")
        time.sleep(3)
    raise RuntimeError(f"snapshot '{name}' did not become active within {timeout:.0f}s")


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


def create_sandbox(
    daytona: Daytona, name: str, role: str, run_id: str, policy: str | None
) -> tuple[Sandbox, str]:
    """A fresh sandbox from the snapshot with the demo's posture: Secret, labels, allow list.

    Returns the sandbox and the network policy it runs behind: the sandbox-level
    allow list when the organization's tier accepts one, otherwise the
    organization policy (the only 400 that is retried is the one saying so).
    Pass the policy a previous sandbox of this run got to skip the probe.
    """
    print(f"=== creating sandbox: {name} (from snapshot) ===")
    params = {
        "snapshot": SNAPSHOT_NAME,
        "name": name,
        # Env var -> Secret name. The sandbox sees the placeholder only.
        "secrets": {"LLM_API_KEY": LLM_VAULT_ENTRY},
        "labels": run_labels(run_id, role),
        "auto_stop_interval": AUTO_STOP_MINUTES,
    }
    if policy == NETWORK_POLICY_ORGANIZATION:
        return daytona.create(CreateSandboxFromSnapshotParams(**params), timeout=180), policy
    try:
        sandbox = daytona.create(
            CreateSandboxFromSnapshotParams(domain_allow_list=DOMAIN_ALLOW_LIST, **params),
            timeout=180,
        )
        return sandbox, NETWORK_POLICY_SANDBOX
    except DaytonaBadRequestError as err:
        if _ORG_POLICY_LOCKED not in str(err):
            raise
    print(
        "    this organization's tier sets network policy for every sandbox; the\n"
        f"    sandbox-level allow list ({DOMAIN_ALLOW_LIST}) was rejected. Running behind\n"
        f"    the {NETWORK_POLICY_ORGANIZATION} instead."
    )
    sandbox = daytona.create(CreateSandboxFromSnapshotParams(**params), timeout=180)
    return sandbox, NETWORK_POLICY_ORGANIZATION


def upload_demo(sandbox: Sandbox, payload: Path) -> str:
    """Create the state and demo directories and upload the payload and the COGX helper."""
    home = sandbox.get_user_home_dir()
    run_in(sandbox, f"mkdir -p {home}/{STATE_DIR} {home}/{DEMO_DIR}/handover-out")
    sandbox.fs.upload_file(payload.read_bytes(), f"{home}/{DEMO_DIR}/{PAYLOAD_NAME}")
    sandbox.fs.upload_file(HELPER.read_bytes(), f"{home}/{DEMO_DIR}/{HELPER_NAME}")
    return home


def create_supervisor(daytona: Daytona, payload: Path, run_id: str) -> tuple[Sandbox, str]:
    """First sandbox in the chain: fresh from the snapshot, payload uploaded, empty memory."""
    sandbox, policy = create_sandbox(
        daytona, f"cognee-supervisor-{run_id}", "supervisor", run_id, policy=None
    )
    upload_demo(sandbox, payload)
    return sandbox, policy


# How the memory travels between two sandboxes of the chain.
HANDOVER_COGX = "cogx"  # export the shared dataset as a COGX archive, restore it in a fresh sandbox
HANDOVER_FORK = "fork"  # copy-on-write clone of the parent VM (Linux VM sandboxes only)
HANDOVER_COPY = "copy"  # archive the whole state out of the parent, into a fresh sandbox
_FORK_UNSUPPORTED = "Forking is not supported"
STATE_ARCHIVE = "/tmp/cognee-handover.tgz"
COGX_DIR = "/tmp/cognee-handover.cogx"
COGX_ARCHIVE = f"{COGX_DIR}.tar.gz"


def fork_into(
    parent: Sandbox, name: str, role: str, run_id: str, policy: str, created: list[Sandbox]
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
    # rather than assume it is inherited. (The allow list only where the tier
    # accepted one at creation; an organization policy is not ours to set.)
    child.update_secrets({"LLM_API_KEY": LLM_VAULT_ENTRY})
    if policy == NETWORK_POLICY_SANDBOX:
        child.update_network_settings(domain_allow_list=DOMAIN_ALLOW_LIST)
    child.set_autostop_interval(AUTO_STOP_MINUTES)
    child.set_labels(run_labels(run_id, role))
    retire(parent)
    return child


def retire(sandbox: Sandbox) -> None:
    """Stop a sandbox whose phase is done and whose state has been handed over.

    A sandbox lives for one phase. Stopping it right away keeps the mounted
    Secret from sitting in a running VM for the rest of the run, and keeps the
    run inside the organization's running-memory quota (three 4 GiB sandboxes
    exceed the 10 GiB a Tier 1/2 organization gets). Cleanup still deletes it.
    """
    sandbox.stop(timeout=120)
    print(f"    {sandbox.name} stopped (its phase is done; state handed over)")


def copy_into(
    daytona: Daytona,
    parent: Sandbox,
    name: str,
    role: str,
    run_id: str,
    policy: str,
    created: list[Sandbox],
) -> Sandbox:
    """Hand the memory over the way the Docker demo does with `sbx cp`: as an archive.

    The parent's cognee state and demo directory (payload + token) are tarred,
    pulled to the host, and pushed into a fresh sandbox created from the
    snapshot with the same posture. Nothing else crosses: the child is a new
    VM with only that archive in it.
    """
    print(f"=== copying {parent.name} -> {name} (the memory handover) ===")
    home = parent.get_user_home_dir()
    run_in(parent, f"tar czf {STATE_ARCHIVE} -C {home} {STATE_DIR} {DEMO_DIR}")
    archive = parent.fs.download_file(STATE_ARCHIVE) or b""
    run_in(parent, f"rm -f {STATE_ARCHIVE}")
    print(f"    state archive: {len(archive) / 1e6:.1f} MB")
    retire(parent)  # before the child exists: the quota counts running sandboxes
    child, _ = create_sandbox(daytona, name, role, run_id, policy)
    created.append(child)
    child_home = child.get_user_home_dir()
    child.fs.upload_file(archive, STATE_ARCHIVE)
    run_in(child, f"tar xzf {STATE_ARCHIVE} -C {child_home} && rm -f {STATE_ARCHIVE}")
    return child


def cogx_into(
    daytona: Daytona,
    parent: Sandbox,
    name: str,
    role: str,
    run_id: str,
    policy: str,
    created: list[Sandbox],
    payload: Path,
) -> Sandbox:
    """Hand the memory over as a COGX archive: export the shared dataset, restore it elsewhere.

    In the parent, ``cogx_handover.py export`` runs ``cognee.export(format="cogx",
    include_permissions=True)`` on the dataset named in the token, as its owner:
    the graph (entities, facts, chunks as raw nodes) plus the owner and ACL
    grants. The packed archive and the token are pulled to the host and pushed
    into a fresh sandbox, where ``cogx_handover.py import`` restores the dataset
    under the archived owner (``cognee.remember(COGXArchiveSource)``, no LLM)
    and re-issues the token with the restored dataset's id. The private dataset
    stays behind: it is not in the archive.
    """
    print(f"=== exporting {parent.name} -> {name} (the memory handover, COGX) ===")
    home = parent.get_user_home_dir()
    token_path = f"{home}/{TOKEN_REL_PATH}"
    run_in(
        parent,
        f"python3 {HELPER_NAME} export --as {SUPERVISOR_EMAIL} "
        f"--token handover-out/handover_token.json --out {COGX_DIR}",
        cwd=f"{home}/{DEMO_DIR}",
        env=cognee_env(home),
    )
    archive = parent.fs.download_file(COGX_ARCHIVE) or b""
    token = parent.fs.download_file(token_path) or b""
    run_in(parent, f"rm -rf {COGX_DIR} {COGX_ARCHIVE}")
    retire(parent)  # before the child exists: the quota counts running sandboxes

    child, _ = create_sandbox(daytona, name, role, run_id, policy)
    created.append(child)
    child_home = upload_demo(child, payload)
    child.fs.upload_file(token, f"{child_home}/{TOKEN_REL_PATH}")
    child.fs.upload_file(archive, COGX_ARCHIVE)
    print(f"=== importing into {name} (COGX restore) ===")
    run_in(
        child,
        f"python3 {HELPER_NAME} import --archive {COGX_ARCHIVE} "
        "--token handover-out/handover_token.json",
        cwd=f"{child_home}/{DEMO_DIR}",
        env=cognee_env(child_home),
    )
    run_in(child, f"rm -f {COGX_ARCHIVE}")
    return child


def handover(
    daytona: Daytona,
    parent: Sandbox,
    name: str,
    role: str,
    run_id: str,
    policy: str,
    mode: str,
    created: list[Sandbox],
    payload: Path,
) -> tuple[Sandbox, str]:
    """Move the memory from `parent` into a new sandbox; returns it and the mode used.

    `cogx` (the default) exports the shared dataset and restores it in a fresh
    sandbox. `auto` tries a fork first. Daytona forks Linux VM sandboxes only;
    a container sandbox — which is what any snapshot built with the declarative
    builder gives, and what organizations without VM runners get — answers
    422 "Forking is not supported", and the handover falls back to the
    archive copy. The mode that worked is returned so the next hop skips the
    failed attempt.
    """
    if mode == HANDOVER_COGX:
        return cogx_into(daytona, parent, name, role, run_id, policy, created, payload), mode
    if mode in ("auto", HANDOVER_FORK):
        try:
            return fork_into(parent, name, role, run_id, policy, created), HANDOVER_FORK
        except DaytonaError as err:
            if mode == HANDOVER_FORK or _FORK_UNSUPPORTED not in str(err):
                raise
            print(
                "    fork not available: this sandbox is a container, not a Linux VM\n"
                "    (fork needs a VM-class snapshot and VM runners in the region).\n"
                "    Handing the memory over as an archive copy instead."
            )
    return copy_into(daytona, parent, name, role, run_id, policy, created), HANDOVER_COPY


def run_phase(sandbox: Sandbox, phase: str) -> None:
    print(f"\n=== sandbox: {sandbox.name} (phase: {phase}) ===")
    home = sandbox.get_user_home_dir()
    run_in(
        sandbox,
        f"python3 {PAYLOAD_NAME} --phase {phase} --token-file handover-out/handover_token.json",
        cwd=f"{home}/{DEMO_DIR}",
        env=cognee_env(home),
    )


# Two ways out of the sandbox. Daytona sets HTTP(S)_PROXY to its egress proxy,
# which is where Secrets are substituted; a client that ignores those
# variables (the way aiohttp did before cognee #5196) connects directly. The
# wall has to hold on both paths, so every probe runs twice: once through the
# proxy, once with proxy handling disabled.
PATHS = (
    # (label, urllib handler expression, goes through the proxy)
    ("proxy path", "u.ProxyHandler()", True),
    ("direct path (proxy env ignored)", "u.ProxyHandler({})", False),
)


def _egress_probe(handler: str) -> str:
    """Exit 0 when the host off the allow list answers, non-zero when it is blocked."""
    return (
        'python3 -c "import urllib.request as u;'
        f"u.build_opener({handler}).open('{EGRESS_PROBE_URL}',timeout=5)\""
    )


def _credential_probe(handler: str) -> str:
    """Print the HTTP status of a call to the LLM host with $LLM_API_KEY as it is in the VM.

    200 would mean the value in the VM authenticates by itself, i.e. it is the
    real key. Expected: 401 (the placeholder is not a key) or blocked.
    """
    return (
        'python3 -c "import os,urllib.request as u,urllib.error as e\n'
        f"r=u.Request('https://{LLM_HOST}/v1/models',"
        "headers={'Authorization':'Bearer '+os.environ['LLM_API_KEY']})\n"
        f"try: print(u.build_opener({handler}).open(r,timeout=8).status)\n"
        "except e.HTTPError as err: print(err.code)\n"
        "except Exception as err: print('blocked', type(err).__name__)\""
    )


def check_boundaries(sandbox: Sandbox, placeholder: str, policy: str, strict: bool) -> list[str]:
    """Prove the isolation properties from inside the sandbox; fail the run if one is missing.

    A wall that does not hold must not end in "round trip passed": every
    verdict is printed, then the first breach raises and the caller's cleanup
    still runs. One case is downgraded to a warning unless `strict`: behind an
    organization tier policy (Tier 1/2), Daytona's egress proxy forwards to
    hosts the firewall blocks on the direct path — the policy is enforced on
    direct connections only. That is Daytona's wall, not one this script can
    set, so it is reported loudly and the run goes on; `--strict-walls` makes
    it a failure. Returns the warnings so the summary can repeat them.
    """
    print(f"\n=== security check inside {sandbox.name} (network wall: {policy}) ===")
    breaches: list[str] = []
    warnings: list[str] = []

    # Read the value without `run_in` (which echoes command output): if the
    # Secret binding failed, `seen` would be the real key. Report the verdict
    # and the length only, never the value.
    seen = (sandbox.process.exec("printf '%s' \"$LLM_API_KEY\"", timeout=20).result or "").strip()
    if seen == placeholder:
        print("    credential wall: LLM_API_KEY inside the sandbox is the placeholder")
    else:
        verdict = f"LLM_API_KEY is NOT the placeholder ({len(seen)} chars): Secret binding failed"
        print(f"    credential wall: {verdict}")
        breaches.append(verdict)

    for label, handler, via_proxy in PATHS:
        status = (sandbox.process.exec(_credential_probe(handler), timeout=30).result or "").strip()
        if status == "200" and not via_proxy:
            verdict = (
                f"{LLM_HOST} accepted the value held in the VM on the {label}: it is a real key"
            )
            print(f"    credential wall: {verdict}")
            breaches.append(verdict)
        elif status == "200":
            print(f"    credential wall: {LLM_HOST} via {label}: 200 (key substituted in transit)")
        else:
            print(
                f"    credential wall: {LLM_HOST} via {label}: {status} (placeholder is not a key)"
            )

    for label, handler, via_proxy in PATHS:
        probe = sandbox.process.exec(_egress_probe(handler), timeout=20)
        if probe.exit_code:
            print(f"    network wall: {EGRESS_PROBE_URL} via {label}: blocked")
            continue
        verdict = f"{EGRESS_PROBE_URL} via {label}: REACHABLE, {policy} not enforced"
        if policy == NETWORK_POLICY_ORGANIZATION and via_proxy and not strict:
            print(f"    network wall: WARNING {verdict}")
            print(
                "        Daytona Tier 1/2 enforces the organization policy on direct connections\n"
                "        only; its egress proxy is not policed. Pass --strict-walls to fail on this."
            )
            warnings.append(verdict)
        else:
            print(f"    network wall: {verdict}")
            breaches.append(verdict)

    if breaches:
        raise RuntimeError(f"isolation breached in {sandbox.name}: " + "; ".join(breaches))
    return warnings


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
    parser.add_argument(
        "--strict-walls",
        action="store_true",
        help="fail on the Tier 1/2 proxy-path egress gap instead of warning about it",
    )
    parser.add_argument(
        "--handover",
        choices=(HANDOVER_COGX, "auto", HANDOVER_FORK, HANDOVER_COPY),
        default=HANDOVER_COGX,
        help="how memory moves between sandboxes: cogx (export the shared dataset as a COGX "
        "archive, restore it in a fresh sandbox; default), copy (archive the whole cognee "
        "state out of the parent), fork (Linux VM sandboxes only), or auto (fork, then copy)",
    )
    args = parser.parse_args()

    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is required (https://app.daytona.io)")
    if not args.payload.is_file():
        raise SystemExit(f"payload not found: {args.payload}")
    if not HELPER.is_file():
        raise SystemExit(f"COGX helper not found: {HELPER}")

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
        supervisor, policy = create_supervisor(daytona, args.payload, run_id)
        sandboxes.append(supervisor)
        run_phase(supervisor, "brief")
        wall_warnings = check_boundaries(supervisor, placeholder, policy, args.strict_walls)

        worker, mode = handover(
            daytona,
            supervisor,
            f"cognee-worker-{run_id}",
            "worker",
            run_id,
            policy,
            args.handover,
            sandboxes,
            args.payload,
        )
        run_phase(worker, "work")

        review, mode = handover(
            daytona,
            worker,
            f"cognee-review-{run_id}",
            "supervisor-review",
            run_id,
            policy,
            mode,
            sandboxes,
            args.payload,
        )
        run_phase(review, "review")

        token = review.fs.download_file(f"{review.get_user_home_dir()}/{TOKEN_REL_PATH}")
        out = Path(__file__).resolve().parent / "handover-out"
        out.mkdir(exist_ok=True)
        (out / "handover_token.json").write_bytes(token or b"")

        print("\nHandover round trip passed across three Daytona sandboxes.")
        print(f"Handover:     {mode}")
        print(f"Network wall: {policy}")
        for warning in wall_warnings:
            print(f"  WARNING {warning}")
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
            # Children first: a forked parent cannot be deleted while a fork
            # of it is still alive.
            for sandbox in reversed(sandboxes):
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
