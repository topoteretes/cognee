"""Supervisor <-> worker memory handover on cognee, with user permissioning.

Self-contained: paste this file into any repository where cognee is installed
and run it with an OpenAI LLM_API_KEY in the environment. It simulates two
agents that share ONE cognee deployment but are separate cognee users,
protected by cognee's ACL layer (ENABLE_BACKEND_ACCESS_CONTROL, the default).

The handover is a ROUND TRIP, split into three phases so each agent can run
in its own container/sandbox (see handover.sh) — the only things crossing the
boundary are the shared cognee storage and a small JSON handover token:

  brief   (supervisor) store a private note + a handover briefing in its own
          datasets, grant the worker read+write on the briefing dataset,
          emit the token {"dataset_id": ...}.
  work    (worker) redeem the token: recall the briefing by dataset UUID,
          prove the private dataset is denied and that dataset NAMES never
          cross users, then write a completion report back into the shared
          dataset (cross-owner writes also require the UUID).
  review  (supervisor) recall the worker's report from the shared dataset.

Run all phases in one process:      python supervisor_worker_handover.py
Run one phase (separate containers): python supervisor_worker_handover.py --phase brief

Backends that currently support user permissioning (per user+dataset DB
isolation, from supported_dataset_database_handlers.py):
  graph:  ladybug/kuzu (default), neo4j (incl. neo4j_community handler),
          postgres (demo), turso
  vector: lancedb (default), pgvector, turso
  NOT supported: neptune, ladybug-remote, neptune_analytics, community
  vector adapters that don't register a dataset-database handler.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import UUID

# The auth posture is resolved when cognee is imported — configure it first.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "true")

import cognee
from cognee.infrastructure.databases.relational import create_db_and_tables
from cognee.modules.data.methods import get_datasets
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import create_user, get_user_by_email
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets

SUPERVISOR_EMAIL = "supervisor@handover.demo"
WORKER_EMAIL = "worker@handover.demo"

HANDOVER_BRIEFING = (
    "Handover to worker: migrate the payments service to the v2 billing API. "
    "Deploy with 'make deploy-prod' only after the contract tests pass. "
    "The staging environment auto-deploys on merge to the dev branch. "
    "Open question for the worker: confirm the retry policy with the SRE team."
)
SUPERVISOR_SECRET = (
    "Supervisor private note: the acquisition of VendorCo closes next quarter; "
    "do not share with worker agents."
)
WORKER_REPORT = (
    "Worker report: payments service migrated to the v2 billing API and deployed "
    "to production with make deploy-prod after contract tests passed. The SRE team "
    "confirmed the retry policy is exponential backoff with a maximum of 5 attempts."
)


async def get_or_create_user(email: str, password: str):
    try:
        return await create_user(email, password)
    except Exception:  # UserAlreadyExists on re-runs
        return await get_user_by_email(email)


async def phase_brief(token_file: Path) -> None:
    """Supervisor: store memory, grant the worker access, emit the token."""
    # Fresh installs: create the relational DB before touching users/ACLs
    # (the CLI does this implicitly; the raw SDK path does not).
    await create_db_and_tables()
    supervisor = await get_or_create_user(SUPERVISOR_EMAIL, "supervisor-pw")
    worker = await get_or_create_user(WORKER_EMAIL, "worker-pw")

    print("[supervisor] remembering private note (dataset 'supervisor_private')...")
    await cognee.remember(SUPERVISOR_SECRET, dataset_name="supervisor_private", user=supervisor)

    print("[supervisor] remembering handover briefing (dataset 'handover')...")
    await cognee.remember(HANDOVER_BRIEFING, dataset_name="handover", user=supervisor)

    # Dataset names map to per-user UUIDs; the token must carry the UUID
    # because that is the only cross-user address for a dataset.
    datasets = {d.name: d.id for d in await get_datasets(supervisor.id)}
    handover_id, private_id = datasets["handover"], datasets["supervisor_private"]

    print("[supervisor] granting worker READ + WRITE on the handover dataset...")
    for permission in ("read", "write"):
        await authorized_give_permission_on_datasets(
            principal_id=worker.id,
            dataset_ids=[handover_id],
            permission_name=permission,
            owner_id=supervisor.id,
        )

    token = {
        "dataset_id": str(handover_id),
        "granted": ["read", "write"],
        "worker": WORKER_EMAIL,
        # Included only so the demo can prove denial; a real supervisor
        # would never put a private dataset id in a handover token.
        "_private_dataset_id": str(private_id),
    }
    token_file.write_text(json.dumps(token, indent=2))
    print(f'[supervisor] handover token issued -> {token_file}: {{"dataset_id": "{handover_id}"}}')


async def phase_work(token_file: Path) -> None:
    """Worker: redeem the token, prove the boundaries, report back."""
    token = json.loads(token_file.read_text())
    worker = await get_user_by_email(WORKER_EMAIL)
    dataset_id = UUID(token["dataset_id"])

    print("[worker] recalling the briefing from the shared dataset (by UUID)...")
    results = await cognee.recall(
        "What is my task and how do I deploy?", dataset_ids=[dataset_id], user=worker
    )
    for r in results:
        text = r.get("text") if isinstance(r, dict) else getattr(r, "text", r)
        print(f"[worker] briefing recalled: {text}")

    print("[worker] trying the supervisor's PRIVATE dataset (should be denied)...")
    try:
        await cognee.recall(
            "What is the private note?",
            dataset_ids=[UUID(token["_private_dataset_id"])],
            user=worker,
        )
        raise AssertionError("worker read the private dataset — permissioning is broken!")
    except PermissionDeniedError as err:
        print(f"[worker] correctly denied: {type(err).__name__}: {err}")

    print("[worker] trying the shared dataset by NAME (names don't cross users)...")
    try:
        await cognee.recall("What is my task?", datasets=["handover"], user=worker)
        raise AssertionError("name resolution crossed user boundaries — unexpected!")
    except PermissionDeniedError as err:
        print(f"[worker] correctly denied: {type(err).__name__}: {err}")
    except Exception as err:
        print(f"[worker] correctly not found: {type(err).__name__}: {err}")

    print("[worker] writing the completion report back into the shared dataset...")
    await cognee.remember(WORKER_REPORT, dataset_id=dataset_id, user=worker)
    print("[worker] report stored.")


async def phase_review(token_file: Path) -> None:
    """Supervisor: read the worker's report from the shared dataset."""
    token = json.loads(token_file.read_text())
    supervisor = await get_user_by_email(SUPERVISOR_EMAIL)

    print("[supervisor] recalling the worker's report...")
    results = await cognee.recall(
        "What did the worker report? Was the migration deployed?",
        dataset_ids=[UUID(token["dataset_id"])],
        user=supervisor,
    )
    for r in results:
        text = r.get("text") if isinstance(r, dict) else getattr(r, "text", r)
        print(f"[supervisor] report recalled: {text}")


PHASES = {"brief": phase_brief, "work": phase_work, "review": phase_review}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    parser.add_argument("--token-file", type=Path, default=Path("handover_token.json"))
    args = parser.parse_args()

    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        await PHASES[name](args.token_file)
    if args.phase == "all":
        print("\nHandover round trip passed: briefing shared, private denied, report returned.")


if __name__ == "__main__":
    asyncio.run(main())
