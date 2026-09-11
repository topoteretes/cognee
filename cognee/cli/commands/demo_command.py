import argparse
import asyncio
import os
from importlib import resources

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException

_DEMO_ARCHIVE = "demo_graph"
_DEMO_DATASET = "demo"

# Queries the bundled graph is known to answer; quickstart.txt names them too.
_DEMO_QUERIES = [
    "Who works at Anthropic?",
    "What does cognee depend on?",
]


def _resolve_archive_path() -> str:
    """Absolute path of the bundled COGX demo archive.

    ``importlib.resources.files`` survives both editable and wheel installs —
    the same pattern the ``remember --sample-data`` fixture uses.
    """
    return str(resources.files("cognee.cli.samples").joinpath(_DEMO_ARCHIVE))


def _result_lines(results) -> list:
    """Flatten CHUNKS_LEXICAL search results into printable text lines.

    Lexical results are chunk payload dicts (or lists of them, per dataset);
    pull the human-readable ``text`` field and fall back to ``str``.
    """
    lines = []
    stack = list(results or [])
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = item + stack
        elif isinstance(item, dict):
            text = item.get("text") or item.get("search_result") or ""
            if isinstance(text, (list, dict)):
                stack = [text] + stack
            elif text:
                lines.append(str(text).strip())
        elif item:
            lines.append(str(item).strip())
    return lines


class DemoCommand(SupportsCliCommand):
    command_string = "demo"
    help_string = "Load a bundled demo knowledge graph and search it — no API key needed"
    docs_url = DEFAULT_DOCS_URL
    description = """
Load a small pre-built knowledge graph and run a first search in seconds,
with zero LLM cost and no API key.

The bundled COGX archive was produced by cognifying the quickstart sample
text once at build time. `demo` restores it graph-only (no embeddings) and
queries it with CHUNKS_LEXICAL — a keyword (BM25) search that needs neither
an LLM nor an embedding provider.

This is proof of life before you risk your own documents and API key:

  cognee-cli demo                       # import + run two example searches
  cognee-cli demo --query "Where does Alice live?"
  cognee-cli demo --dataset-name my_demo

LLM-powered answers (GRAPH_COMPLETION, the default everywhere else) still
require LLM_API_KEY — the demo prints the exact next steps.

Clean up with: cognee-cli forget --dataset demo
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-name",
            "-d",
            default=_DEMO_DATASET,
            help=f"Dataset to load the demo graph into (default: {_DEMO_DATASET})",
        )
        parser.add_argument(
            "--query",
            "-q",
            help="Run this single query instead of the built-in example queries",
        )
        parser.add_argument(
            "--top-k",
            "-k",
            type=int,
            default=3,
            help="Maximum number of results per query (default: 3)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee
            from cognee.modules.migration.sources.cogx_archive import COGXArchiveSource
            from cognee.modules.search.types import SearchType

            # The demo's contract is zero LLM calls. With the defaults
            # (CACHING/AUTO_FEEDBACK on), every answered search fires one
            # turn-analysis LLM call, which fails noisily — or hangs on a dead
            # local endpoint — on exactly the keyless machines this command
            # exists for. Pin it off for this process; set after `import
            # cognee` so the .env load (override=True) cannot clobber it.
            os.environ["AUTO_FEEDBACK"] = "false"

            archive_path = _resolve_archive_path()
            fmt.echo("Loading the bundled demo knowledge graph (no API key required)...")

            async def run_demo():
                try:
                    result = await cognee.remember(
                        COGXArchiveSource(archive_path, mode="preserve"),
                        dataset_name=args.dataset_name,
                        # Graph-only restore: no embeddings, no LLM, no API key.
                        index_vectors=False,
                    )

                    queries = [args.query] if args.query else _DEMO_QUERIES
                    answers = []
                    for query in queries:
                        results = await cognee.search(
                            query_text=query,
                            query_type=SearchType.CHUNKS_LEXICAL,
                            datasets=[args.dataset_name],
                            top_k=args.top_k,
                        )
                        answers.append((query, _result_lines(results)))
                    return result, answers
                except Exception as e:
                    raise CliCommandInnerException(f"Demo failed: {str(e)}") from e

            result, answers = asyncio.run(run_demo())

            items = getattr(result, "items", None) or []
            summary = items[0] if items else {}
            fmt.success(
                f"Demo graph loaded into dataset '{args.dataset_name}' "
                f"({summary.get('graph_nodes', 0)} nodes, "
                f"{summary.get('graph_edges', 0)} edges)."
            )

            for query, lines in answers:
                fmt.echo("")
                fmt.echo(f"Query: {query}")
                if lines:
                    for index, line in enumerate(lines, 1):
                        # The bundled archive is built with small chunks
                        # (tools/build_demo_archive.py), so a whole chunk IS
                        # the answer: print it in full, flattened to one line.
                        fmt.echo(f"  {index}. {' '.join(line.split())}")
                else:
                    fmt.warning("  No results found.")

            fmt.echo("")
            fmt.echo(
                "This demo uses keyword search (CHUNKS_LEXICAL) — it needs no LLM and "
                "no embeddings. LLM answers over your own data need LLM_API_KEY set."
            )
            fmt.echo(
                f'Next: cognee-cli search "your question" -t CHUNKS_LEXICAL -d {args.dataset_name}'
            )
            fmt.echo(
                'Then: set LLM_API_KEY and run cognee-cli remember "<path-or-text>" '
                "to build memory from your own data."
            )
            fmt.echo(f"Clean up with: cognee-cli forget --dataset {args.dataset_name}")
        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error during demo: {str(e)}", error_code=1) from e
