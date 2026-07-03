import argparse
import asyncio
import json

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import OUTPUT_FORMAT_CHOICES
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
        from cognee.cli.config import get_search_type_choices

        parser.add_argument("query_text", help="Your question or search query")
        parser.add_argument(
            "--query-type",
            "-t",
            choices=get_search_type_choices(),
            default="GRAPH_COMPLETION",
            metavar="TYPE",
            help=(
                "Search mode (default: GRAPH_COMPLETION). Common: GRAPH_COMPLETION, "
                "RAG_COMPLETION, CHUNKS, SUMMARIES, TEMPORAL, FEELING_LUCKY."
            ),
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
        from cognee.cli import ui
        from cognee.cli.hints import record_event
        from cognee.cli.preflight import PreflightError, needs_for_search_type, run_preflight
        from cognee.cli.render import render_results, render_session_entries

        try:
            import cognee
            from cognee.modules.search.types import SearchType

            # Session-only mode: -s without -d and without explicit -t
            session_only = (
                args.session_id is not None
                and not args.datasets
                and args.query_type == "GRAPH_COMPLETION"  # i.e., the user didn't pass -t
            )

            caps = ui.detect_caps()

            if not session_only:
                need_llm, need_embeddings = needs_for_search_type(args.query_type)
                run_preflight(need_llm=need_llm, need_embeddings=need_embeddings)

                state, dataset_name, doc_count = asyncio.run(self._memory_state(args))
                if state == "empty":
                    ui.guide_block(
                        "Your memory is empty — nothing has been added yet.",
                        [
                            "cognee-cli add <file, folder, or text>",
                            "cognee-cli cognify",
                            f'cognee-cli recall "{args.query_text}"',
                        ],
                        caps=caps,
                    )
                    return
                if state == "not_cognified":
                    described = (
                        f"{dataset_name} has {doc_count} document(s)"
                        if dataset_name
                        else "Your data is added"
                    )
                    ui.guide_block(
                        f"{described} but no knowledge graph yet.",
                        ["cognee-cli cognify", f'cognee-cli recall "{args.query_text}"'],
                        caps=caps,
                    )
                    return

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

            spinner_text = f"Searching session '{args.session_id}'" if session_only else "Recalling"
            with ui.spinner_line(spinner_text, caps=caps) as spinner:
                results = asyncio.run(run_recall())
                elapsed = spinner.elapsed

            if results:
                record_event("recall_success")

            if args.output_format == "json":
                fmt.echo(json.dumps(results, indent=2, default=str))
                return
            if args.output_format == "simple":
                for i, result in enumerate(results, 1):
                    fmt.echo(f"{i}. {result}")
                return

            is_session = bool(
                results and isinstance(results[0], dict) and results[0].get("_source") == "session"
            )
            if is_session:
                render_session_entries(results, caps=caps)
            else:
                render_results(
                    results,
                    query_type=args.query_type,
                    output_format=args.output_format,
                    elapsed=elapsed,
                    caps=caps,
                )

        except Exception as e:
            # PreflightError must reach the central handler intact so it
            # renders as the calm what/why/fix block, not a generic error.
            if isinstance(e, (CliCommandException, PreflightError)):
                raise
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error recalling: {str(e)}", error_code=1) from e

    async def _memory_state(self, args: argparse.Namespace):
        try:
            from cognee.cli.empty_state import check_memory_state
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            return await check_memory_state(user)
        except Exception:
            return "ready", None, 0
