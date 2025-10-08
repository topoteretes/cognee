import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli.config import SEARCH_TYPE_CHOICES


class SaveCommand(SupportsCliCommand):
    command_string = "save"
    help_string = "Export dataset summaries and search insights to markdown files"
    docs_url = DEFAULT_DOCS_URL
    description = """
Export per-dataset, per-file markdown reports with summaries, ASCII path, question ideas,
and search results. Creates an index.md per dataset.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--datasets",
            nargs="*",
            help="Dataset names to process (default: all accessible)",
        )
        parser.add_argument(
            "--export-root",
            default=None,
            help="Export root directory (default: <data_root_directory>/memory_export)",
        )
        parser.add_argument(
            "--path",
            default=None,
            help="Alias for --export-root",
        )
        parser.add_argument(
            "--max-questions",
            type=int,
            default=10,
            help="Maximum number of question ideas per file (default: 10)",
        )
        parser.add_argument(
            "--search-types",
            nargs="*",
            choices=SEARCH_TYPE_CHOICES,
            default=["GRAPH_COMPLETION", "INSIGHTS", "CHUNKS"],
            help="Search types to run per question",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Top-k results to retrieve for each search (default: 5)",
        )
        parser.add_argument(
            "--no-summary",
            action="store_true",
            help="Exclude file summary section",
        )
        parser.add_argument(
            "--no-path",
            action="store_true",
            help="Exclude ASCII path section",
        )
        parser.add_argument(
            "--concurrency",
            type=int,
            default=4,
            help="Max concurrent files processed per dataset (default: 4)",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="Optional per-dataset timeout in seconds",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            fmt.echo("Starting save export...")

            async def run_save():
                try:
                    result = await cognee.save(
                        datasets=args.datasets,
                        export_root_directory=args.export_root or args.path,
                        max_questions=args.max_questions,
                        search_types=args.search_types,
                        top_k=args.top_k,
                        include_summary=(not args.no_summary),
                        include_ascii_tree=(not args.no_path),
                        concurrency=args.concurrency,
                        timeout=args.timeout,
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to save: {str(e)}")

            results = asyncio.run(run_save())

            if results:
                fmt.success("Export complete:")
                for ds_id, path in results.items():
                    fmt.echo(f"- {ds_id}: {path}")
            else:
                fmt.note("No datasets to export or no outputs generated.")

        except CliCommandInnerException as e:
            fmt.error(str(e))
            raise CliCommandException(self.docs_url)
        except Exception as e:
            fmt.error(f"Unexpected error: {str(e)}")
            raise CliCommandException(self.docs_url)
