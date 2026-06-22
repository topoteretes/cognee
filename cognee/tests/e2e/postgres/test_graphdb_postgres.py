"""End-to-end graph database test using the Postgres backend.

Requires:
  - A running Postgres instance
  - DB_PROVIDER=postgres with valid DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD, DB_NAME
  - LLM_API_KEY set for cognify
"""

import asyncio
from cognee.tests.e2e.postgres.test_graphdb_shared import run_graph_db_test


async def main():
    await run_graph_db_test("postgres")


if __name__ == "__main__":
    asyncio.run(main())
