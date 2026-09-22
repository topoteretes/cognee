"""Run cognee entirely on Turso, the Rust rewrite of SQLite: add -> cognify -> search.

Every layer runs on the Turso rewrite engine (``pyturso``) through cognee's ``turso``
provider: the relational store (users, datasets, pipeline runs), the graph store
(graph-as-tables), the vector store (exact cosine search) and the session cache, each a
local database file under the chosen system directory.

Three phases, so persistence across a restart is visible:

    uv run python examples/guides/turso_local_example.py ingest   # add + cognify + search
    uv run python examples/guides/turso_local_example.py verify   # new process: search only
    uv run python examples/guides/turso_local_example.py cleanup  # forget everything + remove files

Requires: ``pip install cognee"[turso]"`` and an LLM key (``LLM_API_KEY``). The three
providers and ``CACHE_BACKEND`` are forced to ``turso`` by this script (overriding ``.env``), so no ``.env``
changes are needed; the database files live under ``TURSO_EXAMPLE_ROOT`` (default:
``.turso_example`` next to this file).
Set ``TURSO_JOURNAL_MODE=mvcc`` to run the same flow with Turso's concurrent writes.
"""

import asyncio
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(
    os.environ.get("TURSO_EXAMPLE_ROOT", pathlib.Path(__file__).parent / ".turso_example")
)

# The storage roots must be known before cognee is imported (its logging reads them).
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(ROOT / "data"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(ROOT / "system"))

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402
from cognee.shared.logging_utils import ERROR, setup_logging  # noqa: E402

# Importing cognee loads .env with override=True, so the providers are forced here,
# after the import: this example is about running every layer on Turso, whatever the
# .env says. The database configs are read lazily, on first use.
for variable in ("DB_PROVIDER", "GRAPH_DATABASE_PROVIDER", "VECTOR_DB_PROVIDER", "CACHE_BACKEND"):
    os.environ[variable] = "turso"
for variable in ("DB_TURSO_URL", "DB_TURSO_AUTH_TOKEN", "GRAPH_DATABASE_KEY", "VECTOR_DB_URL"):
    os.environ.pop(variable, None)  # remote settings are rejected; use local files

# The cache config is already built during cognee's import; drop the cached
# instances so every settings class re-reads the environment set above.
from cognee.infrastructure.databases.cache.config import get_cache_config  # noqa: E402
from cognee.infrastructure.databases.graph.config import get_graph_config  # noqa: E402
from cognee.infrastructure.databases.relational.config import get_relational_config  # noqa: E402
from cognee.infrastructure.databases.vector.config import get_vectordb_config  # noqa: E402

for cached_config in (
    get_cache_config,
    get_graph_config,
    get_relational_config,
    get_vectordb_config,
):
    cached_config.cache_clear()

DATASET = "turso_example"
TEXT = """
Turso is a rewrite of SQLite in Rust. It keeps the SQLite file format and SQL dialect,
adds multi-version concurrency control so several writers can commit at the same time,
and ships vector distance functions for similarity search. cognee stores its relational
metadata, its knowledge graph and its embeddings in Turso database files.
"""
QUESTION = "What does Turso add on top of SQLite?"


async def show_engines() -> None:
    """Print the engine each layer runs on; ``turso_version()`` exists only on the rewrite."""
    from sqlalchemy import text

    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    relational = get_relational_engine()
    async with relational.engine.connect() as connection:
        version = (await connection.execute(text("SELECT turso_version()"))).scalar()
        journal = (await connection.execute(text("PRAGMA journal_mode"))).scalar()
    print(f"relational: {type(relational).__name__} on Turso {version} (journal_mode={journal})")
    print(f"graph:      {type(await get_graph_engine()).__name__}")
    print(f"vector:     {type(await get_vector_engine_async()).__name__}")
    from cognee.infrastructure.databases.cache import get_cache_engine

    cache = get_cache_engine()
    print(f"cache:      {type(cache).__name__} on {cache.db_uri.split('://')[0]}")


async def search() -> None:
    results = await cognee.search(
        query_text=QUESTION, query_type=SearchType.GRAPH_COMPLETION, datasets=[DATASET]
    )
    print(f"\nQ: {QUESTION}")
    for result in results:
        print(f"A: {result}")
    chunks = await cognee.search(
        query_text="concurrency control", query_type=SearchType.CHUNKS, datasets=[DATASET]
    )
    print(f"\n{len(chunks)} chunk(s) matched a vector search for 'concurrency control'.")


async def ingest() -> None:
    # add() creates the relational database (and its directory) on first use, so the
    # engines are inspected after it, exactly as any cognee flow would see them.
    await cognee.add(TEXT, dataset_name=DATASET)
    await show_engines()
    await cognee.cognify(datasets=[DATASET])
    print("\nadd + cognify done; graph, vectors and metadata are in", ROOT)
    await search()


async def verify() -> None:
    """Runs in a fresh process: nothing is ingested, the stored files must answer."""
    await show_engines()
    await search()


async def cleanup() -> None:
    await cognee.forget(everything=True)
    await cognee.wait_for_background_tasks()
    print("forgot everything")


def main() -> None:
    setup_logging(ERROR)
    phase = sys.argv[1] if len(sys.argv) > 1 else "ingest"
    if phase == "cleanup":
        asyncio.run(cleanup())
        shutil.rmtree(ROOT, ignore_errors=True)
        print("removed", ROOT)
        return
    asyncio.run({"ingest": ingest, "verify": verify}[phase]())


if __name__ == "__main__":
    main()
