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
Recall information from the knowledge graph or session memory.

When --session-id is provided without --datasets or --query-type,
searches the session cache directly by keyword matching.
Otherwise, this is a memory-oriented alias for `cognee search`.
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
            "--session-id",
            "-s",
            default=None,
            help=(
                "Session ID. When used without -d or -t, searches session "
                "memory directly. Otherwise adds session history to the "
                "search context."
            ),
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

            # Session-only mode: -s without -d and without explicit -t
            session_only = (
                args.session_id is not None
                and not args.datasets
                and args.query_type == "GRAPH_COMPLETION"  # i.e., the user didn't pass -t
            )

            if session_only:
                fmt.echo(f"Searching session '{args.session_id}': '{args.query_text}'")
            else:
                datasets_msg = (
                    f" in datasets {args.datasets}" if args.datasets else " across all datasets"
                )
                fmt.echo(f"Recalling: '{args.query_text}' (type: {args.query_type}){datasets_msg}")

            async def run_recall():
                try:
                    from cognee.cli.user_resolution import resolve_cli_user, scoped_session_id

                    session_kwargs = {}
                    if args.session_id is not None:
                        user = await resolve_cli_user(getattr(args, "user_id", None))
                        sid = scoped_session_id(user.id, args.session_id)
                        session_kwargs["session_id"] = sid

                    if session_only:
                        # Pass query_type=None to trigger session-only search
                        results = await cognee.recall(
                            query_text=args.query_text,
                            top_k=args.top_k,
                            **session_kwargs,
                        )
                    else:
                        query_type = SearchType[args.query_type]
                        recall_kwargs = {
                            "query_text": args.query_text,
                            "query_type": query_type,
                            "datasets": args.datasets,
                            "top_k": args.top_k,
                            "system_prompt_path": (
                                args.system_prompt or "answer_simple_question.txt"
                            ),
                            **session_kwargs,
                        }
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

                # Detect session results by _source tag
                is_session = isinstance(results[0], dict) and results[0].get("_source") == "session"

                if is_session:
                    fmt.echo(f"\nFound {len(results)} session entry(ies):")
                    fmt.echo("=" * 60)
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
                else:
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
