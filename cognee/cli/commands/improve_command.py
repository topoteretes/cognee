import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class ImproveCommand(SupportsCliCommand):
    command_string = "improve"
    help_string = "Enrich and improve the knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Enrich and improve the knowledge graph.

This is a memory-oriented alias for `cognee memify`. It runs enrichment
tasks on an existing knowledge graph to add context, rules, and connections.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-name",
            "-d",
            default="main_dataset",
            help="Dataset name (default: main_dataset)",
        )
        parser.add_argument(
            "--dataset-id",
            help="Dataset UUID (alternative to --dataset-name)",
        )
        parser.add_argument(
            "--node-name",
            nargs="*",
            help="Filter to specific named entities",
        )
        parser.add_argument(
            "--session-ids",
            "-s",
            nargs="+",
            help="Session IDs whose feedback and Q&A content should be bridged into the permanent graph",
        )
        parser.add_argument(
            "--feedback-alpha",
            type=float,
            default=0.1,
            help="Learning rate for feedback weight updates (default: 0.1)",
        )
        parser.add_argument(
            "--background",
            "-b",
            action="store_true",
            help="Run processing in background",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            dataset = args.dataset_id if args.dataset_id else args.dataset_name
            fmt.echo(f"Improving knowledge graph for dataset '{dataset}'...")

            async def run_improve():
                try:
                    from uuid import UUID

                    dataset_arg = UUID(args.dataset_id) if args.dataset_id else args.dataset_name

                    result = await cognee.improve(
                        dataset=dataset_arg,
                        node_name=args.node_name,
                        session_ids=args.session_ids,
                        feedback_alpha=args.feedback_alpha,
                        run_in_background=args.background,
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to improve: {str(e)}") from e

            result = asyncio.run(run_improve())

            if args.background:
                fmt.success("Improvement started in background!")
            else:
                fmt.success("Knowledge graph improved successfully!")

            if result and isinstance(result, dict):
                for ds_id, run_info in result.items():
                    status = getattr(run_info, "status", str(run_info))
                    fmt.echo(f"  Dataset {ds_id}: {status}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error improving: {str(e)}", error_code=1) from e
