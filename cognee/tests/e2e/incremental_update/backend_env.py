"""Environment overrides that point the incremental-update suites at a backend.

The suites default to the local stack (kuzu + lancedb + sqlite). Set
``INCR_TEST_GRAPH_PROVIDER`` and/or ``INCR_TEST_VECTOR_PROVIDER`` (``pgvector``)
to run the same tests against other backends:

    INCR_TEST_GRAPH_PROVIDER=postgres \\
    INCR_TEST_DB_HOST=localhost INCR_TEST_DB_PORT=5432 \\
    INCR_TEST_DB_USERNAME=cognee INCR_TEST_DB_PASSWORD=cognee \\
    INCR_TEST_DB_NAME=cognee_db pytest cognee/tests/e2e/incremental_update/

    INCR_TEST_GRAPH_PROVIDER=neo4j INCR_TEST_GRAPH_URL=bolt://... \\
    INCR_TEST_GRAPH_USER=... INCR_TEST_GRAPH_PASSWORD=... pytest ...

The Postgres graph adapter stores its tables in the relational database, so
selecting it also switches the relational provider to Postgres.
"""

import os


def incremental_test_backend_env() -> dict:
    """Env overrides for the graph, vector and (for Postgres) relational backends."""
    provider = os.environ.get("INCR_TEST_GRAPH_PROVIDER", "kuzu")
    vector_provider = os.environ.get("INCR_TEST_VECTOR_PROVIDER", "lancedb")
    env = {
        "GRAPH_DATABASE_PROVIDER": provider,
        "VECTOR_DB_PROVIDER": vector_provider,
        "DB_PROVIDER": "sqlite",
    }
    for source_key, target_key in [
        ("INCR_TEST_GRAPH_URL", "GRAPH_DATABASE_URL"),
        ("INCR_TEST_GRAPH_USER", "GRAPH_DATABASE_USERNAME"),
        ("INCR_TEST_GRAPH_PASSWORD", "GRAPH_DATABASE_PASSWORD"),
        ("INCR_TEST_GRAPH_NAME", "GRAPH_DATABASE_NAME"),
    ]:
        if os.environ.get(source_key):
            env[target_key] = os.environ[source_key]
    if provider in ("postgres", "postgres_demo"):
        host = os.environ.get("INCR_TEST_DB_HOST", "localhost")
        port = os.environ.get("INCR_TEST_DB_PORT", "5432")
        username = os.environ.get("INCR_TEST_DB_USERNAME", "cognee")
        password = os.environ.get("INCR_TEST_DB_PASSWORD", "cognee")
        name = os.environ.get("INCR_TEST_DB_NAME", "cognee_db")
        # The adapter reads GRAPH_DATABASE_* when all of them are set (shared mode)
        # and always in per-dataset mode, so the graph must point at the same
        # server and database as the relational side or it lands in whatever
        # ``.env`` names.
        env.update(
            DB_PROVIDER="postgres",
            DB_HOST=host,
            DB_PORT=port,
            DB_USERNAME=username,
            DB_PASSWORD=password,
            DB_NAME=name,
            GRAPH_DATABASE_URL="",
            GRAPH_DATABASE_HOST=host,
            GRAPH_DATABASE_PORT=port,
            GRAPH_DATABASE_USERNAME=username,
            GRAPH_DATABASE_PASSWORD=password,
            GRAPH_DATABASE_NAME=name,
        )
    if vector_provider == "pgvector":
        env.update(
            VECTOR_DB_HOST=os.environ.get("INCR_TEST_DB_HOST", "localhost"),
            VECTOR_DB_PORT=os.environ.get("INCR_TEST_DB_PORT", "5432"),
            VECTOR_DB_USERNAME=os.environ.get("INCR_TEST_DB_USERNAME", "cognee"),
            VECTOR_DB_PASSWORD=os.environ.get("INCR_TEST_DB_PASSWORD", "cognee"),
            VECTOR_DB_NAME=os.environ.get("INCR_TEST_DB_NAME", "cognee_db"),
        )
    return env


async def reset_backend_state() -> None:
    """Wipe graph, vector and relational state before a scenario runs.

    The embedded stack gets a fresh temp root per module, but a server-backed
    graph (Postgres) is shared by every module in the run, so each scenario
    starts by pruning. Call it inside the scenario's own event loop.
    """
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
