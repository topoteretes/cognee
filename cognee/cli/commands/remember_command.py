import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import CHUNKER_CHOICES
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class RememberCommand(SupportsCliCommand):
    command_string = "remember"
    help_string = "Add data and build the knowledge graph in one step"
    docs_url = DEFAULT_DOCS_URL
    description = """
Add data and build the knowledge graph in one step.

This combines the `add` and `cognify` commands: data is ingested first,
then automatically processed into a structured knowledge graph.

After completion, use `cognee recall` (or `cognee search`) to query the graph.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "data",
            nargs="+",
            help="Data to add: text content, file paths, file URLs, or S3 paths",
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default="main_dataset",
            help="Dataset name (default: main_dataset)",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            help="Maximum tokens per chunk (auto-calculated if not specified)",
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
            help="Run cognify step in background (add always completes first)",
        )
        parser.add_argument(
            "--chunks-per-batch",
            type=int,
            help="Number of chunks to process per task batch",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            fmt.echo(f"Remembering {len(args.data)} item(s) in dataset '{args.dataset_name}'...")

            async def run_remember():
                try:
                    from cognee.modules.chunking.TextChunker import TextChunker

                    chunker_class = TextChunker
                    if args.chunker == "LangchainChunker":
                        try:
                            from cognee.modules.chunking.LangchainChunker import LangchainChunker

                            chunker_class = LangchainChunker
                        except ImportError:
                            fmt.warning("LangchainChunker not available, using TextChunker")
                    elif args.chunker == "CsvChunker":
                        try:
                            from cognee.modules.chunking.CsvChunker import CsvChunker

                            chunker_class = CsvChunker
                        except ImportError:
                            fmt.warning("CsvChunker not available, using TextChunker")

                    data_to_add = args.data[0] if len(args.data) == 1 else args.data

                    result = await cognee.remember(
                        data=data_to_add,
                        dataset_name=args.dataset_name,
                        chunker=chunker_class,
                        chunk_size=args.chunk_size,
                        chunks_per_batch=args.chunks_per_batch,
                        run_in_background=args.background,
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to remember: {str(e)}") from e

            result = asyncio.run(run_remember())

            if args.background:
                fmt.success("Data ingested and cognification started in background!")
            else:
                fmt.success("Data ingested and knowledge graph built successfully!")

            if result:
                if result.dataset_id:
                    fmt.echo(f"  Dataset ID: {result.dataset_id}")
                if result.items_processed:
                    fmt.echo(f"  Items processed: {result.items_processed}")
                if result.content_hash:
                    fmt.echo(f"  Content hash: {result.content_hash}")
                if result.elapsed_seconds is not None:
                    fmt.echo(f"  Elapsed: {result.elapsed_seconds:.1f}s")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Failed to remember: {str(e)}", error_code=1) from e
