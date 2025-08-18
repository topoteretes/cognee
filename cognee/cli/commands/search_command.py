import argparse
import asyncio
import json
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import SEARCH_TYPE_CHOICES, OUTPUT_FORMAT_CHOICES
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

**INSIGHTS**:
    Structured entity relationships and semantic connections.
    Best for: Understanding concept relationships, knowledge mapping.

**CHUNKS**:
    Raw text segments that match the query semantically.
    Best for: Finding specific passages, citations, exact content.

**SUMMARIES**:
    Pre-generated hierarchical summaries of content.
    Best for: Quick overviews, document abstracts, topic summaries.

**CODE**:
    Code-specific search with syntax and semantic understanding.
    Best for: Finding functions, classes, implementation patterns.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("query_text", help="Your question or search query in natural language")
        parser.add_argument(
            "--query-type",
            "-t",
            choices=SEARCH_TYPE_CHOICES,
            default="GRAPH_COMPLETION",
            help="Search mode (default: GRAPH_COMPLETION for conversational AI responses)",
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
        try:
            # Import cognee here to avoid circular imports
            import cognee
            from cognee.modules.search.types import SearchType

            # Convert string to SearchType enum
            query_type = SearchType[args.query_type]

            datasets_msg = (
                f" in datasets {args.datasets}" if args.datasets else " across all datasets"
            )
            fmt.echo(f"Searching for: '{args.query_text}' (type: {args.query_type}){datasets_msg}")

            # Run the async search function
            async def run_search():
                try:
                    results = await cognee.search(
                        query_text=args.query_text,
                        query_type=query_type,
                        datasets=args.datasets,
                        system_prompt_path=args.system_prompt or "answer_simple_question.txt",
                        top_k=args.top_k,
                    )
                    return results
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to search: {str(e)}")

            results = asyncio.run(run_search())

            # Format and display results
            if args.output_format == "json":
                fmt.echo(json.dumps(results, indent=2, default=str))
            elif args.output_format == "simple":
                for i, result in enumerate(results, 1):
                    fmt.echo(f"{i}. {result}")
            else:  # pretty format
                if not results:
                    fmt.warning("No results found for your query.")
                    return

                fmt.echo(f"\nFound {len(results)} result(s) using {args.query_type}:")
                fmt.echo("=" * 60)

                if args.query_type in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
                    # These return conversational responses
                    for i, result in enumerate(results, 1):
                        fmt.echo(f"{fmt.bold('Response:')} {result}")
                        if i < len(results):
                            fmt.echo("-" * 40)
                elif args.query_type == "CHUNKS":
                    # These return text chunks
                    for i, result in enumerate(results, 1):
                        fmt.echo(f"{fmt.bold(f'Chunk {i}:')} {result}")
                        fmt.echo()
                else:
                    # Generic formatting for other types
                    for i, result in enumerate(results, 1):
                        fmt.echo(f"{fmt.bold(f'Result {i}:')} {result}")
                        fmt.echo()

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error searching: {str(e)}", error_code=1)
