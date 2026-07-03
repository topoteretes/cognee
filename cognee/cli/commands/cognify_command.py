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
7. **Content Summarization**: Creates text summaries for navigation

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
        parser.add_argument(
            "--chunks-per-batch",
            type=int,
            help="Number of chunks to process per task batch (try 50 for large single documents).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        import time as time_module

        from cognee.cli import ui
        from cognee.cli.hints import record_event
        from cognee.cli.preflight import run_preflight

        if getattr(args, "verbose", False):
            import logging

            from cognee.shared.logging_utils import set_console_log_level

            set_console_log_level(logging.INFO)

        run_preflight(need_llm=True, need_embeddings=True)

        try:
            # Import cognee here to avoid circular imports
            import cognee

            # Prepare datasets parameter
            datasets = args.datasets if args.datasets else None
            dataset_label = ", ".join(datasets) if datasets else "all datasets"

            # Teach the empty state instead of failing on "no datasets".
            caps = ui.detect_caps()
            if datasets is None:
                state = asyncio.run(self._memory_state(args))
                if state == "empty":
                    ui.guide_block(
                        "Your memory is empty — nothing has been added yet.",
                        [
                            "cognee-cli add <file, folder, or text>",
                            "cognee-cli cognify",
                            'cognee-cli search "your question"',
                        ],
                        caps=caps,
                    )
                    return

            async def run_cognify():
                try:
                    from cognee.cli.user_resolution import resolve_cli_user

                    user = await resolve_cli_user(getattr(args, "user_id", None))

                    # Import chunker classes here
                    from cognee.modules.chunking.TextChunker import TextChunker

                    chunker_class = TextChunker  # Default
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

                    result = await cognee.cognify(
                        datasets=datasets,
                        user=user,
                        chunker=chunker_class,
                        chunk_size=args.chunk_size,
                        ontology_file_path=args.ontology_file,
                        run_in_background=args.background,
                        chunks_per_batch=getattr(args, "chunks_per_batch", None),
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to cognify: {str(e)}") from e

            if args.background:
                asyncio.run(run_cognify())
                ui.success_line("Cognify started in the background.", caps=caps)
                ui.next_step("cognee-cli datasets status", label="Track it:", caps=caps)
                return

            started = time_module.monotonic()
            with ui.pipeline_progress(
                f"Cognifying {dataset_label}", known_stages=ui.COGNIFY_STAGES, caps=caps
            ) as board:
                result = asyncio.run(run_cognify())
                self._raise_on_errored_runs(result)

            elapsed = ui.format_duration(time_module.monotonic() - started)
            record_event("cognify_success")
            board.finish(
                f"Cognified {dataset_label} in {elapsed}",
                next_command='cognee-cli search "What connects the ideas in my data?"',
            )

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error during cognification: {str(e)}", error_code=1) from e

    async def _memory_state(self, args: argparse.Namespace) -> str:
        try:
            from cognee.cli.empty_state import check_memory_state
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(getattr(args, "user_id", None))
            state, _, _ = await check_memory_state(user)
            return state
        except Exception:
            return "ready"

    @staticmethod
    def _raise_on_errored_runs(result) -> None:
        """Blocking cognify returns per-dataset run info — a run that errored
        must fail the command instead of printing a green success line."""
        if not isinstance(result, dict):
            return
        for run_info in result.values():
            status = getattr(run_info, "status", "")
            if "Errored" in str(status):
                payload = getattr(run_info, "payload", None)
                detail = str(payload) if payload else "pipeline run errored"
                raise CliCommandInnerException(f"Failed to cognify: {detail}")
