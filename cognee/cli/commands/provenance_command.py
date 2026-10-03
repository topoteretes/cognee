"""``cognee-cli provenance``: read the audit-grade provenance ledger.

Read-only. The ledger is written by ingestion (``PROVENANCE_TRACKING=true``)
and by the delete paths (tombstones); this command inspects, verifies and
exports it. In-process the CLI runs as the default (or ``--user-id``) user
with direct DB access, so no dataset ACL is applied here — the HTTP surface
(``/api/v1/provenance``) is the one that enforces per-dataset read access.
"""

import argparse
import asyncio
import json
import sys
from typing import Any
from uuid import UUID

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException
from cognee.cli.reference import SupportsCliCommand


def _print_json(payload: Any) -> None:
    fmt.echo(json.dumps(payload, indent=2, default=str))


def _dump_broken_links(links: list[dict[str, Any]], limit: int = 20) -> None:
    for link in links[:limit]:
        reason = link.get("reason", "?")
        line = f"  #{link.get('sequence_id')}  {reason:<18} {link.get('entity_id')}"
        if reason == "chain_break":
            line += (
                f"\n      expected previous {link.get('expected_previous_checksum')}"
                f"\n      stored   previous {link.get('actual_previous_checksum')}"
            )
        fmt.echo(line)
    if len(links) > limit:
        fmt.echo(f"  ... {len(links) - limit} more")


class ProvenanceCommand(SupportsCliCommand):
    command_string = "provenance"
    help_string = "Inspect, verify and export the audit provenance ledger"
    docs_url = DEFAULT_DOCS_URL
    description = """
Read the audit-grade provenance ledger (PROVENANCE_TRACKING=true).

Every graph node and edge cognee writes is recorded as a hash-chained,
append-only row keyed "{dataset_id}:{node_id}" (edges: "rel:{dataset_id}:...").
Deletions never remove rows — they add tombstones. Scope any ledger-wide action
to one dataset with --dataset / --dataset-id.

`verify --anchors` additionally replays the external anchors written by
`anchor` (PROVENANCE_ANCHOR_KEY): it proves the ledger was not rewritten and
re-chained since the last anchor. Exit code 2 means the ledger failed.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="provenance_action", title="actions")

        def add_scope(p: argparse.ArgumentParser) -> None:
            p.add_argument("--dataset", "-d", help="Limit to a dataset by name")
            p.add_argument("--dataset-id", help="Limit to a dataset by UUID")

        def add_format(p: argparse.ArgumentParser) -> None:
            p.add_argument(
                "-f",
                "--format",
                choices=["pretty", "json"],
                default="pretty",
                dest="output_format",
                help="Output format (default: pretty)",
            )

        p_verify = sub.add_parser("verify", help="Verify checksums and the hash chain")
        add_scope(p_verify)
        add_format(p_verify)
        p_verify.add_argument(
            "--anchors",
            action="store_true",
            help="Also verify the ledger against its external anchors (needs the anchor key)",
        )

        p_anchor = sub.add_parser("anchor", help="Sign and record the current chain head")
        add_format(p_anchor)

        p_check = sub.add_parser("check", help="Referential-integrity check")
        add_scope(p_check)
        add_format(p_check)

        p_stats = sub.add_parser("stats", help="Ledger statistics")
        add_scope(p_stats)
        add_format(p_stats)

        p_drift = sub.add_parser("drift", help="Compare ledger snapshots with the graph")
        add_scope(p_drift)
        add_format(p_drift)

        for name, help_text in (
            ("lineage", "Upstream lineage of one entity"),
            ("history", "Version history of one entity"),
            ("entry", "Current ledger row of one entity"),
        ):
            p = sub.add_parser(name, help=help_text)
            p.add_argument(
                "entity_id",
                help='Ledger key, e.g. "{dataset_id}:{node_id}" or "rel:{dataset_id}:..."',
            )
            add_format(p)

        p_export = sub.add_parser("export", help="Export ledger rows as JSON Lines")
        add_scope(p_export)
        p_export.add_argument(
            "-o", "--output", help="Write to this file instead of stdout (.jsonl)"
        )
        p_export.add_argument(
            "--current-only",
            action="store_true",
            help="Skip archived versions (the default export can re-verify history)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "provenance_action", None)
        if not action:
            fmt.error("No action specified. Use --help to see available actions.")
            raise CliCommandException("No action specified", error_code=1)

        handlers = {
            "verify": self._verify,
            "check": self._check,
            "stats": self._stats,
            "drift": self._drift,
            "lineage": self._lineage,
            "history": self._history,
            "entry": self._entry,
            "export": self._export,
            "anchor": self._anchor,
        }
        asyncio.run(handlers[action](args))

    # -- helpers -------------------------------------------------------------- #

    @staticmethod
    async def _resolve_dataset_id(args: argparse.Namespace) -> UUID | None:
        dataset_id = getattr(args, "dataset_id", None)
        dataset = getattr(args, "dataset", None)
        if dataset_id and dataset:
            fmt.error("Provide either --dataset or --dataset-id, not both.")
            raise CliCommandException("Conflicting dataset arguments", error_code=1)
        if dataset_id:
            try:
                return UUID(dataset_id)
            except ValueError as error:
                fmt.error(f"Invalid dataset id: {dataset_id}")
                raise CliCommandException("Invalid dataset id", error_code=1) from error
        if dataset:
            from cognee.cli.user_resolution import resolve_cli_user
            from cognee.modules.data.methods import get_datasets_by_name

            user = await resolve_cli_user(getattr(args, "user_id", None))
            found = await get_datasets_by_name(dataset, user.id)
            if not found:
                fmt.error(f"Dataset '{dataset}' not found.")
                raise CliCommandException("Dataset not found", error_code=1)
            return found[0].id
        return None

    @staticmethod
    def _manager():
        from cognee.modules.provenance import get_provenance_manager

        return get_provenance_manager()

    # -- actions -------------------------------------------------------------- #

    async def _verify(self, args: argparse.Namespace) -> None:
        dataset_id = await self._resolve_dataset_id(args)
        result = await self._manager().verify_chain(dataset_id=dataset_id)
        if getattr(args, "anchors", False):
            result["anchors"] = await self._verify_anchors()
        if args.output_format == "json":
            _print_json(result)
        else:
            scope = f"dataset {dataset_id}" if dataset_id else "whole ledger"
            if result["valid"]:
                fmt.success(f"Ledger verified: {result['total_entries']} rows intact ({scope}).")
            else:
                fmt.error(
                    f"Ledger INVALID: {len(result['broken_links'])} broken link(s) in "
                    f"{result['total_entries']} rows ({scope})."
                )
                _dump_broken_links(result["broken_links"])
            anchors = result.get("anchors")
            if anchors is not None:
                if not anchors["anchored"]:
                    fmt.warning("No external anchors recorded yet (run `provenance anchor`).")
                elif anchors["valid"]:
                    latest = anchors["latest_anchor"]
                    fmt.success(
                        f"{anchors['anchors_checked']} anchor(s) verified; latest at chain "
                        f"position #{latest['sequence_id']} ({latest['anchored_at']})."
                    )
                else:
                    fmt.error(f"{len(anchors['failures'])} anchor failure(s):")
                    for failure in anchors["failures"][:20]:
                        fmt.echo(f"  #{failure.get('sequence_id')}  {failure['reason']}")
        anchors_valid = result.get("anchors", {"valid": True})["valid"]
        if not result["valid"] or not anchors_valid:
            raise CliCommandException("Ledger verification failed", error_code=2)

    async def _verify_anchors(self) -> dict[str, Any]:
        from cognee.modules.provenance.anchors import AnchoringNotConfiguredError

        try:
            return await self._manager().verify_anchors()
        except AnchoringNotConfiguredError as error:
            fmt.error(str(error))
            raise CliCommandException("Anchoring not configured", error_code=1) from error

    async def _anchor(self, args: argparse.Namespace) -> None:
        from cognee.modules.provenance.anchors import AnchoringNotConfiguredError, anchor_path

        try:
            anchor = await self._manager().anchor()
        except AnchoringNotConfiguredError as error:
            fmt.error(str(error))
            raise CliCommandException("Anchoring not configured", error_code=1) from error
        if args.output_format == "json":
            _print_json({"anchor": anchor, "path": anchor_path()})
            return
        if anchor is None:
            fmt.warning("Ledger is empty; nothing to anchor.")
            return
        fmt.success(
            f"Anchored chain head #{anchor['sequence_id']} ({anchor['checksum'][:12]}…) "
            f"to {anchor_path()}"
        )

    async def _check(self, args: argparse.Namespace) -> None:
        dataset_id = await self._resolve_dataset_id(args)
        result = await self._manager().check(dataset_id=dataset_id)
        if args.output_format == "json":
            _print_json(result)
        else:
            if result["valid"]:
                fmt.success(
                    f"References intact: {result['total_entries']} rows, "
                    f"{result['invalidated_count']} live tombstone(s)."
                )
            else:
                fmt.error(f"{result['errors']} dangling reference(s):")
                for ref in result["missing_references"][:20]:
                    fmt.echo(f"  {ref}")
        if not result["valid"]:
            raise CliCommandException("Ledger check failed", error_code=2)

    async def _stats(self, args: argparse.Namespace) -> None:
        dataset_id = await self._resolve_dataset_id(args)
        stats = await self._manager().get_statistics(dataset_id=dataset_id)
        if args.output_format == "json":
            _print_json(stats)
            return
        fmt.echo(f"Scope:            {stats['dataset_id'] or 'whole ledger'}")
        fmt.echo(f"Rows:             {stats['total_entries']}")
        fmt.echo(f"  live:           {stats['live_entries']}")
        fmt.echo(f"  archived:       {stats['archived_entries']}")
        fmt.echo(f"Live tombstones:  {stats['invalidated_count']}")
        fmt.echo(f"Distinct sources: {stats['unique_sources']}")
        if stats["entity_types"]:
            fmt.echo("By type:")
            for entity_type, count in sorted(stats["entity_types"].items()):
                fmt.echo(f"  {entity_type:<16} {count}")

    async def _drift(self, args: argparse.Namespace) -> None:
        dataset_id = await self._resolve_dataset_id(args)
        if dataset_id is None:
            fmt.error("drift needs --dataset or --dataset-id (the graph is read per dataset).")
            raise CliCommandException("Dataset required", error_code=1)
        from cognee.cli.user_resolution import resolve_cli_user
        from cognee.modules.data.methods import get_authorized_dataset

        user = await resolve_cli_user(getattr(args, "user_id", None))
        dataset = await get_authorized_dataset(user, dataset_id, "read")
        if dataset is None:
            fmt.error(f"Dataset {dataset_id} not found or not readable.")
            raise CliCommandException("Dataset not found", error_code=1)

        result = await self._manager().check_drift(dataset_id, dataset.owner_id)
        if args.output_format == "json":
            _print_json(result)
        else:
            if result["valid"]:
                fmt.success(
                    f"Graph matches the ledger: {result['checked']} node(s) checked"
                    + (
                        f", {result['unsnapshotted']} without a snapshot"
                        if result["unsnapshotted"]
                        else ""
                    )
                    + "."
                )
            else:
                fmt.error(
                    f"Drift: {len(result['drifted'])} changed, "
                    f"{len(result['missing_in_graph'])} missing in graph "
                    f"(of {result['checked']} checked)."
                )
                for item in result["drifted"][:20]:
                    fields = ", ".join(item["delta"]) or "?"
                    fmt.echo(f"  changed  {item['entity_id']}  [{fields}]")
                for entity_id in result["missing_in_graph"][:20]:
                    fmt.echo(f"  missing  {entity_id}")
        if not result["valid"]:
            raise CliCommandException("Ledger drift detected", error_code=2)

    async def _lineage(self, args: argparse.Namespace) -> None:
        lineage = await self._manager().get_lineage(args.entity_id)
        if not lineage:
            fmt.error(f"'{args.entity_id}' is not in the ledger.")
            raise CliCommandException("Entity not tracked", error_code=1)
        if args.output_format == "json":
            _print_json(lineage)
            return
        fmt.echo(f"Entity:     {lineage['entity_id']}")
        fmt.echo(f"Entries:    {lineage['entity_count']}")
        fmt.echo(f"Integrity:  {'verified' if lineage['integrity_verified'] else 'BROKEN'}")
        fmt.echo(f"First seen: {lineage['first_seen']}")
        fmt.echo(f"Updated:    {lineage['last_updated']}")
        if lineage["source_documents"]:
            fmt.echo("Sources:")
            for source in lineage["source_documents"]:
                fmt.echo(f"  {source}")
        fmt.echo("Chain:")
        for entry in lineage["lineage_chain"]:
            marker = " (tombstone)" if entry.get("invalidated") else ""
            fmt.echo(f"  {entry['entity_type']:<13} {entry['entity_id']}{marker}")

    async def _history(self, args: argparse.Namespace) -> None:
        history = await self._manager().revision_history(args.entity_id)
        if not history:
            fmt.error(f"'{args.entity_id}' is not in the ledger.")
            raise CliCommandException("Entity not tracked", error_code=1)
        if args.output_format == "json":
            _print_json(history)
            return
        for version in history:
            if version.get("invalidated"):
                fmt.echo(
                    f"v{version['version']}  retracted by {version['invalidated_by']} at "
                    f"{version['recorded_at']}  reason: {version.get('invalidation_reason')}"
                )
            else:
                until = version["valid_until"] or "now"
                fmt.echo(
                    f"v{version['version']}  by {version['author']} at {version['recorded_at']}"
                    f"  valid {version['valid_from']} -> {until}"
                )
                for field, (old, new) in (version.get("delta") or {}).items():
                    fmt.echo(
                        f"      {field}: {json.dumps(old, default=str)} -> {json.dumps(new, default=str)}"
                    )

    async def _entry(self, args: argparse.Namespace) -> None:
        entry = await self._manager().get_provenance(args.entity_id)
        if entry is None:
            fmt.error(f"'{args.entity_id}' is not in the ledger.")
            raise CliCommandException("Entity not tracked", error_code=1)
        _print_json(entry)

    async def _export(self, args: argparse.Namespace) -> None:
        dataset_id = await self._resolve_dataset_id(args)
        rows = self._manager().export(dataset_id=dataset_id, include_archived=not args.current_only)
        count = 0
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                async for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")
                    count += 1
            fmt.success(f"Exported {count} rows to {args.output}")
        else:
            async for row in rows:
                sys.stdout.write(json.dumps(row, default=str) + "\n")
                count += 1
