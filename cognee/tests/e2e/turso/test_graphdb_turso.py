"""End-to-end graph database test using the Turso (libSQL) backend.

Exercises the full add -> cognify -> search pipeline against a local libSQL file.
Requires:
  - LLM_API_KEY set for cognify
  - A working vector database (default LanceDB is fine)

Turso is local and needs no server or extra dependency (a libSQL file is a SQLite
file, read through the aiosqlite driver), so no connection setup is required.
"""

import asyncio
from cognee.tests.e2e.postgres.test_graphdb_shared import run_graph_db_test


async def main():
    await run_graph_db_test("turso")


if __name__ == "__main__":
    asyncio.run(main())
