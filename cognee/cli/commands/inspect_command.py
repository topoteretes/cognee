import argparse
import asyncio
import json
import sys
from uuid import UUID
from datetime import datetime

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


def _format_size(size_bytes: int) -> str:
    if size_bytes is None:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            if size_bytes > 0 and unit != "B":
                return f"{size_bytes:.2f} {unit}"
            else:
                return f"{int(size_bytes)} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


class InspectCommand(SupportsCliCommand):
    command_string = "inspect"
    help_string = "Inspect stored memory (datasets, sessions, counts)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Inspect Cognee memory stores and metrics.

Subcommands:
  overview  Show memory overview (default)
  dataset   Inspect details for a specific dataset
  sessions  List conversation sessions
  recent    List recently ingested items
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--json",
            action="store_true",
            help="Format output as JSON",
        )

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument(
            "--json",
            action="store_true",
            help="Format output as JSON",
        )

        sub = parser.add_subparsers(dest="inspect_action", title="subcommands")

        # overview
        sub.add_parser("overview", parents=[parent_parser], help="Show memory overview")

        # dataset
        p_dataset = sub.add_parser(
            "dataset", parents=[parent_parser], help="Inspect a specific dataset"
        )
        p_dataset.add_argument("name_or_id", help="Dataset name or UUID")
        p_dataset.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of items displayed",
        )

        # sessions
        p_sessions = sub.add_parser(
            "sessions", parents=[parent_parser], help="Inspect conversation sessions"
        )
        p_sessions.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Limit number of sessions displayed",
        )

        # recent
        p_recent = sub.add_parser(
            "recent", parents=[parent_parser], help="Inspect recently ingested items"
        )
        p_recent.add_argument(
            "--limit",
            type=int,
            default=5,
            help="Limit number of items displayed",
        )

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "inspect_action", None)
        if not action or action == "overview":
            self._overview(args)
        elif action == "dataset":
            self._dataset(args)
        elif action == "sessions":
            self._sessions(args)
        elif action == "recent":
            self._recent(args)
        else:
            fmt.error(f"Unknown action: {action}")
            raise CliCommandException(f"Unknown action: {action}", error_code=1)

    async def _get_recent_ingests_helper(self, user, limit: int):
        from cognee.api.v1.datasets.datasets import datasets as datasets_api

        try:
            datasets_list = await datasets_api.list_datasets(user=user)
        except Exception:
            return []

        all_items = []
        for dataset in datasets_list:
            try:
                items = await datasets_api.list_data(dataset.id, user=user)
                for item in items:
                    all_items.append((item, dataset.name))
            except Exception:
                pass

        # Sort by created_at descending in-memory
        all_items.sort(
            key=lambda x: x[0].created_at if x[0].created_at else datetime.min,
            reverse=True,
        )

        # De-duplicate by Data ID
        seen = set()
        deduped = []
        for item, ds_name in all_items:
            if item.id not in seen:
                seen.add(item.id)
                deduped.append((item, ds_name))

        return deduped[:limit]

    def _overview(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.cli.user_resolution import resolve_cli_user
            from cognee.api.v1.datasets.datasets import datasets as datasets_api
            from cognee.modules.session_lifecycle.metrics import list_session_rows
            from cognee.infrastructure.databases.graph import get_graph_engine
            from cognee.context_global_variables import (
                set_database_global_context_variables,
                backend_access_control_enabled,
            )

            user = await resolve_cli_user(getattr(args, "user_id", None))

            try:
                datasets_list = await datasets_api.list_datasets(user=user)
            except Exception:
                datasets_list = []

            # Gather dataset info
            datasets_info = []
            total_documents = 0
            total_size = 0
            total_tokens = 0

            for dataset in datasets_list:
                try:
                    items = await datasets_api.list_data(dataset.id, user=user)
                    item_count = len(items)
                    size_sum = sum(item.data_size for item in items if item.data_size is not None)
                    token_sum = sum(
                        item.token_count for item in items if item.token_count is not None
                    )

                    datasets_info.append(
                        {
                            "id": str(dataset.id),
                            "name": dataset.name,
                            "created_at": dataset.created_at.isoformat()
                            if dataset.created_at
                            else None,
                            "document_count": item_count,
                            "storage_size_bytes": size_sum,
                            "token_count": token_sum,
                        }
                    )
                    total_documents += item_count
                    total_size += size_sum
                    total_tokens += token_sum
                except Exception:
                    pass

            # Gather graph metrics
            total_nodes = 0
            total_edges = 0
            graph_metrics_available = False

            if backend_access_control_enabled():
                for dataset in datasets_list:
                    try:
                        async with set_database_global_context_variables(
                            dataset.id, dataset.owner_id
                        ):
                            graph_engine = await get_graph_engine()
                            metrics = await graph_engine.get_graph_metrics(include_optional=False)
                            total_nodes += metrics.get("num_nodes", 0) or 0
                            total_edges += metrics.get("num_edges", 0) or 0
                            graph_metrics_available = True
                    except Exception:
                        pass
            else:
                try:
                    graph_engine = await get_graph_engine()
                    metrics = await graph_engine.get_graph_metrics(include_optional=False)
                    total_nodes = metrics.get("num_nodes", 0) or 0
                    total_edges = metrics.get("num_edges", 0) or 0
                    graph_metrics_available = True
                except Exception:
                    pass

            # Gather sessions
            try:
                session_page = await list_session_rows(user_id=user.id, limit=1000)
                sessions_count = session_page.total
                sessions_by_status = {"running": 0, "completed": 0, "failed": 0, "abandoned": 0}
                for sess in session_page.sessions:
                    status = sess.effective_status
                    if status in sessions_by_status:
                        sessions_by_status[status] += 1
            except Exception:
                sessions_count = 0
                sessions_by_status = {"running": 0, "completed": 0, "failed": 0, "abandoned": 0}

            # Gather recent ingests
            recent_items = await self._get_recent_ingests_helper(user, limit=5)
            recent_ingests_info = []
            for item, ds_name in recent_items:
                recent_ingests_info.append(
                    {
                        "id": str(item.id),
                        "name": item.name,
                        "dataset_name": ds_name,
                        "mime_type": item.mime_type,
                        "size_bytes": item.data_size,
                        "token_count": item.token_count,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                    }
                )

            if getattr(args, "json", False):
                result_json = {
                    "datasets": datasets_info,
                    "totals": {
                        "datasets_count": len(datasets_list),
                        "documents_count": total_documents,
                        "storage_size_bytes": total_size,
                        "token_count": total_tokens,
                        "graph_nodes_count": total_nodes if graph_metrics_available else "N/A",
                        "graph_edges_count": total_edges if graph_metrics_available else "N/A",
                        "sessions_count": sessions_count,
                    },
                    "sessions_by_status": sessions_by_status,
                    "recent_ingests": recent_ingests_info,
                }
                print(json.dumps(result_json, indent=2))
                return

            # Print human-readable tables
            fmt.echo("\n=== Memory Overview ===")

            # Datasets
            fmt.echo("\nDatasets")
            fmt.echo("-" * 65)
            fmt.echo(f"{'Name':<35} {'Items':<12} {'Size':<15}")
            fmt.echo("-" * 65)
            for ds in datasets_info:
                fmt.echo(
                    f"{ds['name']:<35} {ds['document_count']:<12} {_format_size(ds['storage_size_bytes']):<15}"
                )
            fmt.echo("-" * 65)
            fmt.echo(f"{'Total':<35} {total_documents:<12} {_format_size(total_size):<15}")

            # Sessions
            fmt.echo("\nSessions")
            fmt.echo("-" * 35)
            fmt.echo(f"{'Status':<20} {'Count':<10}")
            fmt.echo("-" * 35)
            for status, count in sessions_by_status.items():
                fmt.echo(f"{status.capitalize():<20} {count:<10}")
            fmt.echo("-" * 35)
            fmt.echo(f"{'Total':<20} {sessions_count:<10}")

            # Graph Metrics
            fmt.echo("\nGraph Statistics")
            fmt.echo("-" * 35)
            fmt.echo(f"Nodes: {total_nodes if graph_metrics_available else 'N/A'}")
            fmt.echo(f"Edges: {total_edges if graph_metrics_available else 'N/A'}")
            fmt.echo("-" * 35)

            # Recent Ingests
            fmt.echo("\nRecent Ingests")
            fmt.echo("-" * 75)
            fmt.echo(f"{'Dataset':<20} {'File':<35} {'Time':<18}")
            fmt.echo("-" * 75)
            for item, ds_name in recent_items:
                time_str = item.created_at.strftime("%Y-%m-%d %H:%M") if item.created_at else "N/A"
                fmt.echo(f"{ds_name:<20} {item.name:<35} {time_str:<18}")
            fmt.echo("-" * 75)
            fmt.echo("")

        asyncio.run(run())

    def _dataset(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.cli.user_resolution import resolve_cli_user
            from cognee.api.v1.datasets.datasets import datasets as datasets_api
            from cognee.modules.data.methods import get_authorized_dataset

            user = await resolve_cli_user(getattr(args, "user_id", None))
            name_or_id = args.name_or_id
            dataset = None

            # Try resolving as UUID
            try:
                dataset_uuid = UUID(name_or_id)
                dataset = await get_authorized_dataset(user, dataset_uuid, "read")
            except ValueError:
                # Not a UUID, resolve by name
                try:
                    datasets_list = await datasets_api.list_datasets(user=user)
                    dataset = next(
                        (d for d in datasets_list if d.name.lower() == name_or_id.lower()), None
                    )
                except Exception:
                    pass

            if not dataset:
                fmt.error(f"Dataset '{name_or_id}' not found or not accessible.")
                raise CliCommandException(
                    f"Dataset '{name_or_id}' not found or not accessible.", error_code=1
                )

            try:
                items = await datasets_api.list_data(dataset.id, user=user)
            except Exception:
                items = []

            # Sort items by created_at descending (latest first)
            items.sort(
                key=lambda x: x.created_at if x.created_at else datetime.min,
                reverse=True,
            )

            total_size = sum(item.data_size for item in items if item.data_size is not None)
            total_tokens = sum(item.token_count for item in items if item.token_count is not None)
            last_activity = items[0].created_at if items else None

            # Apply limit if specified
            if args.limit is not None:
                items = items[: args.limit]

            documents_info = []
            for item in items:
                documents_info.append(
                    {
                        "id": str(item.id),
                        "name": item.name,
                        "mime_type": item.mime_type,
                        "size_bytes": item.data_size,
                        "token_count": item.token_count,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                    }
                )

            if getattr(args, "json", False):
                result_json = {
                    "dataset_id": str(dataset.id),
                    "dataset_name": dataset.name,
                    "owner_id": str(dataset.owner_id),
                    "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                    "totals": {
                        "document_count": len(items),
                        "storage_size_bytes": total_size,
                        "token_count": total_tokens,
                    },
                    "documents": documents_info,
                }
                print(json.dumps(result_json, indent=2))
                return

            # Print human-readable tables
            fmt.echo(f"\n=== Dataset: {dataset.name} ===")
            fmt.echo(f"ID: {dataset.id}")
            fmt.echo(f"Total Size: {_format_size(total_size)}")
            fmt.echo(f"Total Tokens: {total_tokens}")
            fmt.echo(
                f"Last Activity: {last_activity.strftime('%Y-%m-%d %H:%M') if last_activity else 'N/A'}"
            )

            fmt.echo("\nDocuments")
            fmt.echo("-" * 95)
            fmt.echo(f"{'ID':<38} {'Name':<25} {'Type':<12} {'Size':<10} {'Tokens':<8}")
            fmt.echo("-" * 95)
            for doc in documents_info:
                doc_name = doc["name"]
                if len(doc_name) > 22:
                    doc_name = doc_name[:19] + "..."
                fmt.echo(
                    f"{doc['id']:<38} {doc_name:<25} {doc['mime_type'][:12]:<12} "
                    f"{_format_size(doc['size_bytes']):<10} {doc['token_count'] or 0:<8}"
                )
            fmt.echo("-" * 95)
            fmt.echo("")

        asyncio.run(run())

    def _sessions(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.cli.user_resolution import resolve_cli_user
            from cognee.modules.session_lifecycle.metrics import list_session_rows

            user = await resolve_cli_user(getattr(args, "user_id", None))

            try:
                session_page = await list_session_rows(
                    user_id=user.id,
                    limit=args.limit or 50,
                )
                sessions_list = session_page.sessions
            except Exception:
                sessions_list = []

            sessions_info = [s.to_dict() for s in sessions_list]

            if getattr(args, "json", False):
                print(json.dumps(sessions_info, indent=2))
                return

            fmt.echo("\n=== Conversation Sessions ===")
            fmt.echo("-" * 80)
            fmt.echo(f"{'Session ID':<30} {'Status':<12} {'Created':<18} {'Tokens (In/Out)':<18}")
            fmt.echo("-" * 80)
            for s in sessions_info:
                # Convert timestamps
                created_str = "N/A"
                if s.get("started_at"):
                    try:
                        dt = datetime.fromisoformat(s["started_at"])
                        created_str = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        created_str = str(s["started_at"])[:16]

                tokens_str = f"{s.get('tokens_in', 0)}/{s.get('tokens_out', 0)}"
                sess_id = s["session_id"]
                # Strip user prefix from ID for displaying if present
                if ":" in sess_id:
                    sess_id = sess_id.split(":", 1)[1]
                if len(sess_id) > 28:
                    sess_id = sess_id[:25] + "..."

                fmt.echo(
                    f"{sess_id:<30} {s['effective_status']:<12} {created_str:<18} {tokens_str:<18}"
                )
            fmt.echo("-" * 80)
            fmt.echo("")

        asyncio.run(run())

    def _recent(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            recent_items = await self._get_recent_ingests_helper(user, limit=args.limit or 5)

            recent_info = []
            for item, ds_name in recent_items:
                recent_info.append(
                    {
                        "id": str(item.id),
                        "name": item.name,
                        "dataset_name": ds_name,
                        "mime_type": item.mime_type,
                        "size_bytes": item.data_size,
                        "token_count": item.token_count,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                    }
                )

            if getattr(args, "json", False):
                print(json.dumps(recent_info, indent=2))
                return

            fmt.echo("\n=== Recently Ingested Items ===")
            fmt.echo("-" * 90)
            fmt.echo(f"{'Dataset':<20} {'File':<35} {'Size':<12} {'Time':<18}")
            fmt.echo("-" * 90)
            for item, ds_name in recent_items:
                time_str = item.created_at.strftime("%Y-%m-%d %H:%M") if item.created_at else "N/A"
                file_name = item.name
                if len(file_name) > 32:
                    file_name = file_name[:29] + "..."
                fmt.echo(
                    f"{ds_name:<20} {file_name:<35} "
                    f"{_format_size(item.data_size):<12} {time_str:<18}"
                )
            fmt.echo("-" * 90)
            fmt.echo("")

        asyncio.run(run())
