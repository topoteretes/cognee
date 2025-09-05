import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import CHUNKER_CHOICES
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class CognifyCommand(SupportsCliCommand):
    command_string = "cognify"
    help_string = "Transform ingested data into a structured knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Transform ingested data into a structured knowledge graph.

This is the core processing step in Cognee that converts raw text and documents
into an intelligent knowledge graph. It analyzes content, extracts entities and
relationships, and creates semantic connections for enhanced search and reasoning.

Processing Pipeline:
1. **Document Classification**: Identifies document types and structures
2. **Permission Validation**: Ensures user has processing rights  
3. **Text Chunking**: Breaks content into semantically meaningful segments
4. **Entity Extraction**: Identifies key concepts, people, places, organizations
5. **Relationship Detection**: Discovers connections between entities
6. **Graph Construction**: Builds semantic knowledge graph with embeddings
7. **Content Summarization**: Creates hierarchical summaries for navigation

After successful cognify processing, use `cognee search` to query the knowledge graph.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--datasets",
            "-d",
            nargs="*",
            help="Dataset name(s) to process. Processes all available data if not specified. Can be multiple: --datasets dataset1 dataset2",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            help="Maximum tokens per chunk. Auto-calculated based on LLM if not specified (~512-8192 tokens)",
        )
        parser.add_argument(
            "--ontology-file", help="Path to RDF/OWL ontology file for domain-specific entity types"
        )
        parser.add_argument(
            "--chunker",
            choices=CHUNKER_CHOICES,
            default="TextChunker",
            help="Text chunking strategy (default: TextChunker)",
        )
        parser.add_argument(
            "--background",
            "-b",
            action="store_true",
            help="Run processing in background and return immediately (recommended for large datasets)",
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Show detailed progress information"
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            # Import cognee here to avoid circular imports
            import cognee

            # Prepare datasets parameter
            datasets = args.datasets if args.datasets else None
            dataset_msg = f" for datasets {datasets}" if datasets else " for all available data"
            fmt.echo(f"Starting cognification{dataset_msg}...")

            if args.verbose:
                fmt.note("This process will analyze your data and build knowledge graphs.")
                fmt.note("Depending on data size, this may take several minutes.")
                if args.background:
                    fmt.note(
                        "Running in background mode - the process will continue after this command exits."
                    )

            # Prepare chunker parameter - will be handled in the async function

            # Run the async cognify function
            async def run_cognify():
                try:
                    # Import chunker classes here
                    from cognee.modules.chunking.TextChunker import TextChunker

                    chunker_class = TextChunker  # Default
                    if args.chunker == "LangchainChunker":
                        try:
                            from cognee.modules.chunking.LangchainChunker import LangchainChunker

                            chunker_class = LangchainChunker
                        except ImportError:
                            fmt.warning("LangchainChunker not available, using TextChunker")

                    result = await cognee.cognify(
                        datasets=datasets,
                        chunker=chunker_class,
                        chunk_size=args.chunk_size,
                        ontology_file_path=args.ontology_file,
                        run_in_background=args.background,
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to cognify: {str(e)}")

            result = asyncio.run(run_cognify())

            if args.background:
                fmt.success("Cognification started in background!")
                if args.verbose and result:
                    fmt.echo(
                        "Background processing initiated. Use pipeline monitoring to track progress."
                    )
            else:
                fmt.success("Cognification completed successfully!")
                if args.verbose and result:
                    fmt.echo(f"Processing results: {result}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error during cognification: {str(e)}", error_code=1)
