import argparse
import asyncio
import json

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
            default=None,
            help="Search mode (default: auto-route, or session search when -s is used without -d)",
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
            "--session-id",
            "-s",
            default=None,
            help="Session ID to include conversation history in the search context",
        )
        parser.add_argument(
            "--output-format",
            "-f",
            choices=OUTPUT_FORMAT_CHOICES,
            default="pretty",
            help="Output format (default: pretty)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee
            from cognee.modules.search.types import SearchType

            query_type = SearchType[args.query_type] if args.query_type else None
            type_label = args.query_type or "auto"

            datasets_msg = (
                f" in datasets {args.datasets}" if args.datasets else " across all datasets"
            )
            if args.session_id and not args.datasets and query_type is None:
                fmt.echo(f"Searching session '{args.session_id}': '{args.query_text}'")
            else:
                fmt.echo(f"Recalling: '{args.query_text}' (type: {type_label}){datasets_msg}")

            async def run_recall():
                try:
                    from cognee.cli.user_resolution import resolve_cli_user, scoped_session_id

                    session_kwargs = {}
                    if args.session_id is not None:
                        user = await resolve_cli_user(getattr(args, "user_id", None))
                        sid = scoped_session_id(user.id, args.session_id)
                        session_kwargs["session_id"] = sid

                    recall_kwargs = {
                        "query_text": args.query_text,
                        "datasets": args.datasets,
                        "top_k": args.top_k,
                        **session_kwargs,
                    }
                    if query_type is not None:
                        recall_kwargs["query_type"] = query_type
                    if args.system_prompt:
                        recall_kwargs["system_prompt_path"] = args.system_prompt
                    elif query_type is not None:
                        recall_kwargs["system_prompt_path"] = "answer_simple_question.txt"

                    results = await cognee.recall(**recall_kwargs)
                    return results
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to recall: {str(e)}") from e

            results = asyncio.run(run_recall())

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

                if isinstance(results[0], dict) and "question" in results[0]:
                    # Session QA entries
                    for i, entry in enumerate(results, 1):
                        q = entry.get("question", "")
                        a = entry.get("answer", "")
                        t = entry.get("time", "")
                        header = f"[{t}] " if t else ""
                        if q:
                            fmt.echo(f"{fmt.bold(f'{header}Q:')} {q}")
                        if a:
                            fmt.echo(f"{fmt.bold('A:')} {a}")
                        if i < len(results):
                            fmt.echo("-" * 40)
                elif args.query_type in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
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
