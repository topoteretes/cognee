import argparse
import asyncio
import json

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class StatusCommand(SupportsCliCommand):
    command_string = "status"
    help_string = "Show processing status for datasets"
    docs_url = DEFAULT_DOCS_URL
    description = """
Show processing status for datasets.

Returns aggregate counts by default. Use --items to see per-file detail
including error messages and content hashes. Use --since to filter by time.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--datasets",
            "-d",
            nargs="*",
            help="Dataset name(s) to check",
        )
        parser.add_argument(
            "--items",
            action="store_true",
            default=False,
            help="Show per-item detail instead of aggregates",
        )
        parser.add_argument(
            "--since",
            help="Only include items created after this ISO timestamp",
        )
        parser.add_argument(
            "--output-format",
            "-f",
            choices=["pretty", "json"],
            default="pretty",
            help="Output format (default: pretty)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee
            from datetime import datetime

            since = None
            if args.since:
                since = datetime.fromisoformat(args.since)

            async def run_status():
                try:
                    return await cognee.status(
                        datasets=args.datasets,
                        items=args.items,
                        since=since,
                    )
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to get status: {str(e)}") from e

            results = asyncio.run(run_status())

            if args.output_format == "json":
                data = [r.model_dump(mode="json") for r in results]
                fmt.echo(json.dumps(data, indent=2, default=str))
            else:
                if not results:
                    fmt.warning("No datasets found.")
                    return

                if args.items:
                    for item in results:
                        status_str = item.status.upper()
                        line = f"  {item.name}: {status_str}"
                        if item.error:
                            line += f" -- {item.error}"
                        fmt.echo(line)
                else:
                    for ds in results:
                        fmt.echo(
                            f"  {ds.dataset_name}: "
                            f"{ds.completed}/{ds.item_count} completed, "
                            f"{ds.pending} pending"
                        )

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error getting status: {str(e)}", error_code=1) from e
