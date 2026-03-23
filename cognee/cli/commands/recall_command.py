import argparse
import asyncio
import json
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import SEARCH_TYPE_CHOICES, OUTPUT_FORMAT_CHOICES
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class RecallCommand(SupportsCliCommand):
    command_string = "recall"
    help_string = "Recall information from the knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Recall information from the knowledge graph.

This is a memory-oriented alias for `cognee search`. All search types and
options are supported.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("query_text", help="Your question or search query")
        parser.add_argument(
            "--query-type",
            "-t",
            choices=SEARCH_TYPE_CHOICES,
            default="GRAPH_COMPLETION",
            help="Search mode (default: GRAPH_COMPLETION)",
        )
        parser.add_argument(
            "--datasets",
            "-d",
            nargs="*",
            help="Dataset name(s) to search within",
        )
        parser.add_argument(
            "--top-k",
            "-k",
            type=int,
            default=10,
            help="Maximum number of results (default: 10)",
        )
        parser.add_argument(
            "--system-prompt",
            help="Custom system prompt file for LLM-based search types",
        )
        parser.add_argument(
            "--output-format",
            "-f",
            choices=OUTPUT_FORMAT_CHOICES,
            default="pretty",
            help="Output format (default: pretty)",
        )

    def execute(self, args: argparse.Namespace) -> Optional[dict]:
        try:
            import cognee
            from cognee.modules.search.types import SearchType

            query_type = SearchType[args.query_type]

            datasets_msg = (
                f" in datasets {args.datasets}" if args.datasets else " across all datasets"
            )
            fmt.echo(f"Recalling: '{args.query_text}' (type: {args.query_type}){datasets_msg}")

            async def run_recall():
                try:
                    results = await cognee.recall(
                        query_text=args.query_text,
                        query_type=query_type,
                        datasets=args.datasets,
                        system_prompt_path=args.system_prompt or "answer_simple_question.txt",
                        top_k=args.top_k,
                    )
                    return results
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to recall: {str(e)}") from e

            results = asyncio.run(run_recall())

            # In JSON mode, return structured data and skip display logic
            if fmt.is_json_mode():
                return {
                    "results": results,
                    "query_type": args.query_type,
                    "count": len(results) if results else 0,
                }

            if args.output_format == "json":
                fmt.echo(json.dumps(results, indent=2, default=str))
            elif args.output_format == "simple":
                for i, result in enumerate(results, 1):
                    fmt.echo(f"{i}. {result}")
            else:
                if not results:
                    fmt.warning("No results found for your query.")
                    return

                fmt.echo(f"\nFound {len(results)} result(s) using {args.query_type}:")
                fmt.echo("=" * 60)

                if args.query_type in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
                    for i, result in enumerate(results, 1):
                        fmt.echo(f"{fmt.bold('Response:')} {result}")
                        if i < len(results):
                            fmt.echo("-" * 40)
                elif args.query_type == "CHUNKS":
                    for i, result in enumerate(results, 1):
                        fmt.echo(f"{fmt.bold(f'Chunk {i}:')} {result}")
                        fmt.echo()
                else:
                    for i, result in enumerate(results, 1):
                        fmt.echo(f"{fmt.bold(f'Result {i}:')} {result}")
                        fmt.echo()

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error recalling: {str(e)}", error_code=1) from e
