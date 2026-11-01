"""Move one cognee dataset between sandboxes as a COGX archive. Runs inside the sandbox.

COGX is cognee's exchange format: a directory with ``manifest.json`` plus one
JSONL file per record kind (entities, facts, documents, raw graph nodes),
and, with ``include_permissions=True``, a ``permissions.json`` holding the
dataset's owner and ACL grants with their credentials. ``cognee.export()``
writes it; ``cognee.remember(COGXArchiveSource(dir))`` restores it with no
LLM call (the graph is re-stored as it was; only embeddings are recomputed).

Two commands, one per side of a handover::

    python3 cogx_handover.py export --as supervisor@handover.demo \\
        --token handover-out/handover_token.json --out /tmp/handover.cogx
    python3 cogx_handover.py import --archive /tmp/handover.cogx.tar.gz \\
        --token handover-out/handover_token.json

``export`` reads the shared dataset's id from the token, exports it as the
given user (who must be allowed to read it), and packs the archive directory
into ``<out>.tar.gz``. ``import`` unpacks, restores the dataset under its
archived owner with the archived grants, and rewrites the token: restored
accounts get new ids and a dataset id is derived from its owner's id, so the
restored dataset has a new id too.
"""

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "true")

import cognee
from cognee.infrastructure.databases.relational import create_db_and_tables
from cognee.modules.data.methods import get_datasets
from cognee.modules.migration.archive import pack_archive, unpack_archive
from cognee.modules.migration.sources.cogx_archive import COGXArchiveSource
from cognee.modules.users.methods import get_user_by_email

# Written next to manifest.json so the importing side knows what it holds.
SIDECAR = "handover.json"


async def export_cmd(as_email: str, token_file: Path, out_dir: Path) -> None:
    token = json.loads(token_file.read_text())
    dataset_id = UUID(token["dataset_id"])
    user = await get_user_by_email(as_email)
    if user is None:
        raise SystemExit(f"no such user here: {as_email}")

    result = await cognee.export(
        dataset_id, format="cogx", destination=out_dir, user=user, include_permissions=True
    )
    (out_dir / SIDECAR).write_text(
        json.dumps({"dataset_name": result.dataset_name, "dataset_id": str(dataset_id)})
    )
    files = sorted(p.name for p in out_dir.iterdir())
    tarball = pack_archive(out_dir, out_dir.with_suffix(out_dir.suffix + ".tar.gz"))
    print(
        f"[cogx] exported dataset '{result.dataset_name}' ({dataset_id}) as {as_email}: "
        f"{result.num_nodes} nodes, {result.num_edges} edges"
    )
    print(f"[cogx] archive files: {', '.join(files)}")
    print(f"[cogx] packed -> {tarball} ({tarball.stat().st_size / 1e3:.0f} kB)")


async def import_cmd(archive: Path, token_file: Path) -> None:
    await create_db_and_tables()  # fresh sandbox: no relational DB yet
    with tempfile.TemporaryDirectory() as tmp, archive.open("rb") as fileobj:
        root = unpack_archive(fileobj, tmp)
        sidecar = json.loads((root / SIDECAR).read_text())
        permissions = json.loads((root / "permissions.json").read_text())
        owner_email = permissions["owner"]["email"]
        grants = {g["user"]["email"]: g["permissions"] for g in permissions.get("grants", [])}

        # The importer is this sandbox's default user (a superuser, created on
        # first use). The archive's owner and granted users are recreated from
        # permissions.json and the dataset is restored under the owner.
        result = await cognee.remember(
            COGXArchiveSource(root), dataset_name=sidecar["dataset_name"]
        )

    summary = next((i for i in result.items if i.get("kind") == "migration_import"), {})
    print(
        f"[cogx] restored dataset '{sidecar['dataset_name']}' under owner {owner_email}: "
        f"{summary.get('graph_nodes', '?')} nodes, {summary.get('graph_edges', '?')} edges, "
        f"status {result.status}"
    )
    for email, permissions_ in grants.items():
        print(f"[cogx] grant restored: {email} -> {', '.join(permissions_)}")

    owner = await get_user_by_email(owner_email)
    restored = next(d for d in await get_datasets(owner.id) if d.name == sidecar["dataset_name"])
    token = json.loads(token_file.read_text())
    old_id = token["dataset_id"]
    if str(restored.id) != old_id:
        token["dataset_id"] = str(restored.id)
        token_file.write_text(json.dumps(token, indent=2))
        print(
            f"[cogx] token re-issued: dataset {old_id} is {restored.id} here "
            "(restored accounts have new ids, and a dataset id derives from its owner's)"
        )
    else:
        print(f"[cogx] token unchanged: dataset id {old_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--as", dest="as_email", required=True, help="export as this cognee user")
    exp.add_argument("--token", type=Path, required=True, help="handover token (dataset_id)")
    exp.add_argument("--out", type=Path, required=True, help="archive directory to create")
    imp = sub.add_parser("import")
    imp.add_argument("--archive", type=Path, required=True, help="packed .tar.gz archive")
    imp.add_argument("--token", type=Path, required=True, help="token to re-issue")
    args = parser.parse_args()

    if args.command == "export":
        asyncio.run(export_cmd(args.as_email, args.token, args.out))
    else:
        asyncio.run(import_cmd(args.archive, args.token))


if __name__ == "__main__":
    main()
