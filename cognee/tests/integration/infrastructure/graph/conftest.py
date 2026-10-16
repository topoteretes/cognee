"""Fixtures shared by the graph adapter integration tests."""

from __future__ import annotations

import logging

import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False


def _postgres_graph_url_from_env():
    """Postgres graph connection URL when the configured backend is postgres, else None.

    Driven entirely by ``.env`` (which CI sets): the postgres graph adapter talks
    to the relational engine's database, so the postgres contract params run
    exactly when ``GRAPH_DATABASE_PROVIDER=postgres`` and ``DB_PROVIDER=postgres``
    are configured, and skip on any other stack (keeps kuzu/sqlite CI green).
    """
    from cognee.infrastructure.databases.graph.config import get_graph_config

    if get_graph_config().graph_database_provider not in ("postgres", "postgres_demo"):
        return None
    from cognee.infrastructure.databases.relational import get_relational_engine

    return get_relational_engine().db_uri


async def _make_postgres_adapter():
    """Fresh-schema Postgres graph adapter, or skip when ``.env`` isn't postgres."""
    url = _postgres_graph_url_from_env()
    if not url:
        pytest.skip("postgres graph backend not configured (set GRAPH_DATABASE_PROVIDER=postgres)")

    from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
    from cognee.infrastructure.databases.graph.postgres_demo.tables import _meta

    adapter = PostgresDemoAdapter(url)
    try:
        async with adapter.engine.begin() as conn:
            await conn.run_sync(_meta.drop_all)
        await adapter.initialize()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Ignoring exception in _make_postgres_adapter", exc_info=True)
        await adapter.close()
        pytest.skip(f"postgres graph backend not reachable: {exc}")
    return adapter


async def _make_neo4j_adapter():
    """Fresh (fully wiped) Neo4j graph adapter, or skip when ``.env`` isn't neo4j.

    Driven entirely by ``.env`` (which CI sets): runs exactly when
    ``GRAPH_DATABASE_PROVIDER=neo4j`` is configured and skips on any other stack.
    Each case starts from an empty graph, so the target Neo4j must be a
    disposable test instance — the fixture wipes every node before yielding.
    """
    from cognee.infrastructure.databases.graph.config import get_graph_config

    config = get_graph_config()
    if config.graph_database_provider.lower() != "neo4j":
        pytest.skip("neo4j graph backend not configured (set GRAPH_DATABASE_PROVIDER=neo4j)")
    if not config.graph_database_url:
        pytest.skip("neo4j graph backend URL not configured")

    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    adapter = Neo4jAdapter(
        graph_database_url=config.graph_database_url,
        graph_database_username=config.graph_database_username or None,
        graph_database_password=config.graph_database_password or None,
        graph_database_name=config.graph_database_name or None,
    )
    try:
        await adapter.initialize()
        await adapter.query("MATCH (n) DETACH DELETE n")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Ignoring exception in _make_neo4j_adapter", exc_info=True)
        await adapter.close()
        pytest.skip(f"neo4j graph backend not reachable: {exc}")
    return adapter


@pytest_asyncio.fixture(params=["ladybug", "postgres", "neo4j"])
async def graph_provenance_adapter(request, tmp_path):
    if request.param == "ladybug":
        if not HAS_LADYBUG:
            pytest.skip("ladybug not installed")
        adapter = LadybugAdapter(str(tmp_path / "graph_db"))
    elif request.param == "postgres":
        adapter = await _make_postgres_adapter()
    elif request.param == "neo4j":
        adapter = await _make_neo4j_adapter()
    else:
        raise AssertionError(f"Unknown graph provenance provider: {request.param}")

    try:
        yield adapter
    finally:
        await adapter.close()
