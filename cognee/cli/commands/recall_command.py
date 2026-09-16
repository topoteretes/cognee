import argparse
import asyncio
import json

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.code_search import (
    add_code_arguments,
    build_code_query,
    handle_diagram_out,
)
from cognee.cli.config import (
    AUTO_QUERY_TYPE,
    OUTPUT_FORMAT_CHOICES,
    SEARCH_TYPE_CHOICES,
)
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli.hints import hint_recall_empty
from cognee.cli.recall_output import print_recall_results
from cognee.cli.reference import SupportsCliCommand


class RecallCommand(SupportsCliCommand):
    command_string = "recall"
    help_string = "Recall information from the knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Recall information from the knowledge graph or session memory.

When --session-id is provided without --datasets or --query-type,
searches the session cache directly by keyword matching.
Otherwise, this is a memory-oriented alias for `cognee search`.

Without --query-type the query is auto-routed to a search strategy by a
rule-based classifier (no LLM call); HYBRID_COMPLETION is the fallback.
Pass --query-type to pin one. See docs/recall-vs-search.md.

With --query-type CODE, --code-query selects the code-graph operation and
--diagram / --diagram-out draw the result (Mermaid or Graphviz).
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("query_text", help="Your question or search query")
        parser.add_argument(
            "--query-type",
            "-t",
            choices=SEARCH_TYPE_CHOICES,
            default=None,
            help="Search mode. Omit to auto-route the query; pass a value to pin one.",
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
        add_code_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee
            from cognee.modules.search.types import SearchType

            code_query = build_code_query(args, args.query_type)

            # `-d` with no names parses to [], which recall() would treat as an
            # empty dataset list rather than "all datasets".
            args.datasets = args.datasets or None

            # Session-only mode: -s without -d and without explicit -t. Only the
            # echo line differs; recall() decides the sources from the arguments.
            session_only = (
                args.session_id is not None and not args.datasets and args.query_type is None
            )
            # No -t means "let the SDK route"; the label is refined from the
            # results once the resolved search type is known.
            effective_query_type = args.query_type or AUTO_QUERY_TYPE

            if session_only:
                fmt.echo(f"Searching session '{args.session_id}': '{args.query_text}'")
            else:
                datasets_msg = (
                    f" in datasets {args.datasets}" if args.datasets else " across all datasets"
                )
                fmt.echo(
                    f"Recalling: '{args.query_text}' (type: {effective_query_type}){datasets_msg}"
                )

            async def run_recall():
                try:
                    session_kwargs = {}
                    if args.session_id is not None:
                        session_kwargs["session_id"] = args.session_id

                    recall_kwargs = {
                        "query_text": args.query_text,
                        "datasets": args.datasets,
                        "top_k": args.top_k,
                        "system_prompt_path": (args.system_prompt or "answer_simple_question.txt"),
                        **session_kwargs,
                    }
                    if args.query_type is not None:
                        recall_kwargs["query_type"] = SearchType[args.query_type]
                    if code_query is not None:
                        # recall() runs code_query in its dedicated "code"
                        # lane, which the auto scope never implies.
                        recall_kwargs["code_query"] = code_query
                        recall_kwargs["scope"] = ["code"]
                    return await cognee.recall(**recall_kwargs)
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to recall: {e!s}") from e

            results = asyncio.run(run_recall())

            if args.output_format == "json":
                fmt.echo(json.dumps(results, indent=2, default=str))
            elif args.output_format == "simple":
                for i, result in enumerate(results, 1):
                    fmt.echo(f"{i}. {result}")
            else:
                if not results:
                    fmt.warning("No results found for your query.")
                    # Hint scoped to the pretty output path: json/simple are
                    # for scripting so an extra line would corrupt the sink.
                    hint_dataset = (
                        args.datasets[0] if getattr(args, "datasets", None) else "<dataset-name>"
                    )
                    hint_recall_empty(hint_dataset)
                    return

                print_recall_results(results, effective_query_type)

            handle_diagram_out(results, args)

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error recalling: {e!s}", error_code=1) from e
