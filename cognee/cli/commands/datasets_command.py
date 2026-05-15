import argparse
import asyncio
import json
from uuid import UUID

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class DatasetsCommand(SupportsCliCommand):
    command_string = "datasets"
    help_string = "Manage datasets (list, create, inspect, status, delete)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Manage Cognee datasets.

Subcommands:
  list      List all accessible datasets
  create    Create a new dataset
  data      List data items in a dataset
  status    Show processing status for datasets
  graph     Export knowledge graph for a dataset (JSON)
  delete    Delete a dataset by ID
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="datasets_action", title="actions")

        # list
        sub.add_parser("list", help="List all accessible datasets")

        # create
        p_create = sub.add_parser("create", help="Create a new dataset")
        p_create.add_argument("name", help="Dataset name")

        # data
        p_data = sub.add_parser("data", help="List data items in a dataset")
        p_data.add_argument("dataset_id", help="Dataset UUID")

        # status
        p_status = sub.add_parser("status", help="Show processing status")
        p_status.add_argument("dataset_ids", nargs="+", help="One or more dataset UUIDs")
        p_status.add_argument(
            "--pipelines",
            nargs="+",
            default=None,
            help=("Optional pipeline names to check (defaults to: cognify_pipeline)"),
        )

        # graph
        p_graph = sub.add_parser("graph", help="Export knowledge graph (JSON)")
        p_graph.add_argument("dataset_id", help="Dataset UUID")
        p_graph.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")

        # delete
        p_del = sub.add_parser("delete", help="Delete a dataset by ID")
        p_del.add_argument("dataset_id", help="Dataset UUID")
        p_del.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "datasets_action", None)
        if not action:
            fmt.error("No action specified. Use --help to see available actions.")
            raise CliCommandException("No action specified", error_code=1)

        dispatch = {
            "list": self._list,
            "create": self._create,
            "data": self._data,
            "status": self._status,
            "graph": self._graph,
            "delete": self._delete,
        }
        dispatch[action](args)

    def _list(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            ds = await cognee.datasets.list_datasets(user=user)
            if not ds:
                fmt.echo("No datasets found.")
                return
            fmt.echo(f"{'ID':<38} {'Name':<30} {'Created'}")
            fmt.echo("-" * 90)
            for d in ds:
                fmt.echo(f"{str(d.id):<38} {d.name:<30} {d.created_at}")

        asyncio.run(run())

    def _create(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.cli.user_resolution import resolve_cli_user
            from cognee.modules.data.methods import create_dataset, get_datasets_by_name
            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.modules.users.permissions.methods import give_permission_on_dataset

            user = await resolve_cli_user(getattr(args, "user_id", None))
            existing = await get_datasets_by_name([args.name], user.id)
            if existing:
                fmt.echo(f"Dataset '{args.name}' already exists: {existing[0].id}")
                return

            db_engine = get_relational_engine()
            async with db_engine.get_async_session() as session:
                dataset = await create_dataset(dataset_name=args.name, user=user, session=session)
                for perm in ("read", "write", "share", "delete"):
                    await give_permission_on_dataset(user, dataset.id, perm)

            fmt.success(f"Created dataset '{args.name}' ({dataset.id})")

        asyncio.run(run())

    def _data(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            dataset_id = UUID(args.dataset_id)
            items = await cognee.datasets.list_data(dataset_id, user=user)
            if not items:
                fmt.echo("No data items found.")
                return
            fmt.echo(f"{'ID':<38} {'Name':<30} {'Type':<15} {'Created'}")
            fmt.echo("-" * 110)
            for d in items:
                fmt.echo(f"{str(d.id):<38} {d.name:<30} {d.mime_type:<15} {d.created_at}")

        asyncio.run(run())

    def _status(self, args: argparse.Namespace) -> None:
        async def run():
            import cognee

            ids = [UUID(i) for i in args.dataset_ids]
            statuses = await cognee.datasets.get_status(ids, pipeline_names=args.pipelines)
            if not statuses:
                fmt.echo("No status information available.")
                return
            for ds_id, pipeline_statuses in statuses.items():
                if isinstance(pipeline_statuses, dict):
                    if not pipeline_statuses:
                        fmt.echo(f"{ds_id}: <no pipeline runs found>")
                        continue
                    formatted = ", ".join(
                        f"{pipeline}={status}" for pipeline, status in pipeline_statuses.items()
                    )
                    fmt.echo(f"{ds_id}: {formatted}")
                else:
                    fmt.echo(f"{ds_id}: {pipeline_statuses}")

        asyncio.run(run())

    def _graph(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.cli.user_resolution import resolve_cli_user
            from cognee.modules.graph.methods import get_formatted_graph_data

            user = await resolve_cli_user(getattr(args, "user_id", None))
            dataset_id = UUID(args.dataset_id)
            graph = await get_formatted_graph_data(dataset_id, user)

            output = json.dumps(graph, indent=2, default=str)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                fmt.success(f"Graph exported to {args.output}")
            else:
                fmt.echo(output)

        asyncio.run(run())

    def _delete(self, args: argparse.Namespace) -> None:
        dataset_id = UUID(args.dataset_id)
        if not args.force:
            if not fmt.confirm(f"Delete dataset {dataset_id}? This cannot be undone"):
                fmt.echo("Cancelled.")
                return

        async def run():
            import cognee
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            await cognee.datasets.empty_dataset(dataset_id, user=user)
            fmt.success(f"Dataset {dataset_id} deleted.")

        asyncio.run(run())
