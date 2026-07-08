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
        from cognee.cli import ui
        from cognee.cli.hints import record_event
        from cognee.cli.preflight import run_preflight

        run_preflight(need_llm=True, need_embeddings=True)

        try:
            import cognee

            caps = ui.detect_caps()

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

            item_count = len(args.data)
            item_word = "item" if item_count == 1 else "items"
            with ui.pipeline_progress(
                f"Remembering {item_count} {item_word} in {args.dataset_name}",
                known_stages=ui.COGNIFY_STAGES,
                caps=caps,
            ) as board:
                result = asyncio.run(run_remember())
                # remember() reports pipeline failures in the result instead
                # of raising — an errored run must not print a green success.
                if result is not None and getattr(result, "status", "") == "errored":
                    error_detail = getattr(result, "error", None) or "remember run errored"
                    raise CliCommandInnerException(f"Failed to remember: {error_detail}")

            if args.background:
                board.stop()
                ui.success_line("Ingestion started in the background.", caps=caps)
                return

            record_event("remember_success")
            elapsed = ""
            if result is not None and getattr(result, "elapsed_seconds", None) is not None:
                elapsed = f" in {ui.format_duration(result.elapsed_seconds)}"
            processed = ""
            if result is not None and getattr(result, "items_processed", None):
                processed = f" — {result.items_processed} item(s) processed"
            board.finish(
                f"Remembered into {args.dataset_name}{elapsed}{processed}",
                next_command='cognee-cli recall "What do you know about this?"',
            )

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Failed to remember: {str(e)}", error_code=1) from e
