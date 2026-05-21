"""End-to-end test for the pghybrid backend.

Exercises add -> cognify -> search with USE_UNIFIED_PROVIDER=pghybrid,
which uses the PostgresHybridAdapter for both graph and vector operations
in a single Postgres database.

Requires:
  - A running Postgres instance with pgvector extension
  - LLM_API_KEY set for cognify
"""

import os
import asyncio


async def main():
    # Import cognee first -- its __init__ calls dotenv.load_dotenv(override=True)
    # which overwrites os.environ from .env. We must set our overrides AFTER this.
    from cognee.tests.e2e.postgres.test_graphdb_shared import run_graph_db_test

    # Override env vars (after dotenv has run)
    os.environ["USE_UNIFIED_PROVIDER"] = "pghybrid"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["DB_PROVIDER"] = "postgres"
    os.environ["DB_HOST"] = os.environ.get("DB_HOST", "localhost")
    os.environ["DB_PORT"] = os.environ.get("DB_PORT", "5432")
    os.environ["DB_USERNAME"] = os.environ.get("DB_USERNAME", "cognee")
    os.environ["DB_PASSWORD"] = os.environ.get("DB_PASSWORD", "cognee")
    os.environ["DB_NAME"] = os.environ.get("DB_NAME", "cognee_db")

    # Clear all cached configs and engine factories so they re-read env vars
    from cognee.infrastructure.databases.relational.config import get_relational_config
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    get_relational_config.cache_clear()
    get_graph_config.cache_clear()
    create_relational_engine.cache_clear()
    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()

    await run_graph_db_test("postgres")


if __name__ == "__main__":
    asyncio.run(main())
