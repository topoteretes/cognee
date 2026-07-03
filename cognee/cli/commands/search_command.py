import argparse
import asyncio
import json
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import OUTPUT_FORMAT_CHOICES
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class SearchCommand(SupportsCliCommand):
    command_string = "search"
    help_string = "Search and query the knowledge graph for insights, information, and connections"
    docs_url = DEFAULT_DOCS_URL
    description = """
Search and query the knowledge graph for insights, information, and connections.

This is the final step in the Cognee workflow that retrieves information from the
processed knowledge graph. It supports multiple search modes optimized for different
use cases - from simple fact retrieval to complex reasoning and code analysis.

Search Types & Use Cases:

**GRAPH_COMPLETION** (Default - Recommended):
    Natural language Q&A using full graph context and LLM reasoning.
    Best for: Complex questions, analysis, summaries, insights.

**RAG_COMPLETION**:
    Traditional RAG using document chunks without graph structure.
    Best for: Direct document retrieval, specific fact-finding.

**CHUNKS**:
    Raw text segments that match the query semantically.
    Best for: Finding specific passages, citations, exact content.

**SUMMARIES**:
    Pre-generated summaries of content.
    Best for: Quick overviews, document abstracts, topic summaries.

**TEMPORAL**:
    Time-aware graph search.
    Best for: "What happened before/after X?", evolving facts.

**FEELING_LUCKY**:
    Automatically picks the best search type for your query.

All types are listed in --help under --query-type.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        from cognee.cli.config import get_search_type_choices

        parser.add_argument("query_text", help="Your question or search query in natural language")
        parser.add_argument(
            "--query-type",
            "-t",
            choices=get_search_type_choices(),
            default="GRAPH_COMPLETION",
            metavar="TYPE",
            help=(
                "Search mode (default: GRAPH_COMPLETION). Common: GRAPH_COMPLETION, "
                "RAG_COMPLETION, CHUNKS, SUMMARIES, TEMPORAL, FEELING_LUCKY. "
                "Run with an invalid value to list all types."
            ),
        )
        parser.add_argument(
            "--datasets",
            "-d",
            nargs="*",
            help="Dataset name(s) to search within. Searches all accessible datasets if not specified",
        )
        parser.add_argument(
            "--top-k",
            "-k",
            type=int,
            default=10,
            help="Maximum number of results to return (default: 10, max: 100)",
        )
        parser.add_argument(
            "--system-prompt",
            help="Custom system prompt file for LLM-based search types (default: answer_simple_question.txt)",
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
        from cognee.cli.preflight import needs_for_search_type, run_preflight
        from cognee.cli.render import render_results

        need_llm, need_embeddings = needs_for_search_type(args.query_type)
        run_preflight(need_llm=need_llm, need_embeddings=need_embeddings)

        try:
            # Import cognee here to avoid circular imports
            import cognee
            from cognee.modules.search.types import SearchType

            # The parser derives choices from the enum, so this cannot fail —
            # guarded anyway so a stale caller gets a calm list, not a KeyError.
            try:
                query_type = SearchType[args.query_type]
            except KeyError:
                valid = ", ".join(t.name for t in SearchType)
                raise CliCommandException(
                    f"unknown search type '{args.query_type}'. Valid types: {valid}",
                    error_code=2,
                ) from None

            caps = ui.detect_caps()

            # Searching an empty memory is a normal first-run moment: teach the
            # rail and exit 0 before any LLM or embedding call runs.
            state, dataset_name, doc_count = asyncio.run(self._memory_state(args))
            if state == "empty":
                ui.guide_block(
                    "Your memory is empty — nothing has been added yet.",
                    [
                        "cognee-cli add <file, folder, or text>",
                        "cognee-cli cognify",
                        f'cognee-cli search "{args.query_text}"',
                    ],
                    caps=caps,
                )
                return
            if state == "not_cognified":
                described = (
                    f"{dataset_name} has {doc_count} document(s)"
                    if dataset_name
                    else ("Your data is added")
                )
                ui.guide_block(
                    f"{described} but no knowledge graph yet.",
                    ["cognee-cli cognify", f'cognee-cli search "{args.query_text}"'],
                    caps=caps,
                )
                return

            # Run the async search function
            async def run_search():
                try:
                    from cognee.cli.user_resolution import resolve_cli_user, scoped_session_id

                    user = await resolve_cli_user(getattr(args, "user_id", None))

                    results = await cognee.search(
                        query_text=args.query_text,
                        query_type=query_type,
                        user=user,
                        datasets=args.datasets,
                        system_prompt_path=args.system_prompt or "answer_simple_question.txt",
                        top_k=args.top_k,
                        session_id=scoped_session_id(user.id),
                    )
                    return results
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to search: {str(e)}") from e

            with ui.spinner_line("Searching your memory", caps=caps) as spinner:
                results = asyncio.run(run_search())
                elapsed = spinner.elapsed

            if results:
                record_event("search_success")
            render_results(
                results,
                query_type=args.query_type,
                output_format=args.output_format,
                elapsed=elapsed,
                caps=caps,
            )

        except Exception as e:
            if isinstance(e, (CliCommandException,)):
                raise
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error searching: {str(e)}", error_code=1) from e

    async def _memory_state(self, args: argparse.Namespace):
        try:
            from cognee.cli.empty_state import check_memory_state
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            return await check_memory_state(user)
        except Exception:
            return "ready", None, 0
