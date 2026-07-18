"""Tests for shared-database (schema-per-dataset) isolation mode.

The ``pgvector_shared`` and ``postgres_graph_shared`` dataset handlers keep each
dataset's tables in a dedicated Postgres *schema* (``ds_<dataset_id>``) inside one
shared database, instead of provisioning a whole database per dataset. Isolation
is enforced by pinning the per-dataset engine's ``search_path`` to that schema.

These tests exercise the real adapters + admin helpers against a running Postgres
(default: cognee:cognee@localhost:5432/cognee_db) and skip if it is unreachable.
The handler-lifecycle test additionally requires cognee to be configured with a
Postgres relational backend (DB_PROVIDER=postgres), since the shared handlers
anchor to the relational configuration; it skips otherwise.
"""

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.postgres import (
    create_pg_schema_if_not_exists,
    drop_pg_schema_if_exists,
    dataset_schema_name,
)


def _db() -> dict:
    return dict(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        username=os.environ.get("DB_USERNAME", "cognee"),
        password=os.environ.get("DB_PASSWORD", "cognee"),
        name=os.environ.get("DB_NAME", "cognee_db"),
    )


def _base_url() -> str:
    d = _db()
    return (
        f"postgresql+asyncpg://{d['username']}:{d['password']}@{d['host']}:{d['port']}/{d['name']}"
    )


class FakeEmbeddingEngine:
    """Deterministic, offline embedder so these tests need no LLM/API key."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    async def embed_text(self, texts):
        return [
            [((abs(hash(t)) % 997) / 997.0) + i * 0.001 for i in range(self.dim)] for t in texts
        ]

    def get_vector_size(self) -> int:
        return self.dim

    def get_batch_size(self) -> int:
        return 100


async def _postgres_reachable() -> bool:
    engine = create_async_engine(_base_url())
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


async def _schema_exists(schema: str) -> bool:
    engine = create_async_engine(_base_url())
    try:
        async with engine.connect() as conn:
            res = await conn.execute(
                text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :s"),
                {"s": schema},
            )
            return res.scalar() is not None
    finally:
        await engine.dispose()


async def _tables_in_schema(schema: str) -> list:
    engine = create_async_engine(_base_url())
    try:
        async with engine.connect() as conn:
            res = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = :s"),
                {"s": schema},
            )
            return [r[0] for r in res.fetchall()]
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def two_schemas():
    """Yield two fresh dataset schema names and drop them on teardown."""
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable for shared-schema isolation tests")
    d = _db()
    s1 = f"ds_{uuid.uuid4().hex}"
    s2 = f"ds_{uuid.uuid4().hex}"
    yield s1, s2
    for s in (s1, s2):
        await drop_pg_schema_if_exists(
            d["name"],
            s,
            host=d["host"],
            port=d["port"],
            username=d["username"],
            password=d["password"],
        )


def test_dataset_schema_name_is_valid_identifier():
    dataset_id = uuid.uuid4()
    name = dataset_schema_name(dataset_id)
    assert name == f"ds_{dataset_id.hex}"
    assert name.isidentifier()
    assert len(name) <= 63  # Postgres identifier limit


@pytest.mark.asyncio
async def test_pgvector_schema_isolation(two_schemas):
    """Two PGVector adapters pinned to different schemas don't see each other."""
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
        IndexSchema,
    )

    d = _db()
    s1, s2 = two_schemas
    for s in (s1, s2):
        await create_pg_schema_if_not_exists(
            d["name"],
            s,
            host=d["host"],
            port=d["port"],
            username=d["username"],
            password=d["password"],
            with_vector_extension=True,
        )

    emb = FakeEmbeddingEngine()
    a1 = PGVectorAdapter(_base_url(), "", emb, schema=s1)
    a2 = PGVectorAdapter(_base_url(), "", emb, schema=s2)
    try:
        await a1.create_data_points(
            "Entity_name",
            [IndexSchema(id=uuid.uuid4(), text="alice"), IndexSchema(id=uuid.uuid4(), text="bob")],
        )
        await a2.create_data_points("Entity_name", [IndexSchema(id=uuid.uuid4(), text="carol")])

        # Same logical collection name lives independently in each schema.
        assert "Entity_name" in await _tables_in_schema(s1)
        assert "Entity_name" in await _tables_in_schema(s2)
        assert "Entity_name" not in await _tables_in_schema("public")

        assert len(await a1.search("Entity_name", query_text="alice", limit=50)) == 2
        assert len(await a2.search("Entity_name", query_text="alice", limit=50)) == 1

        # get_table_names is scoped to the adapter's own schema.
        assert all(s2 not in n for n in await a1.get_table_names())

        # Dropping schema 1 leaves schema 2 fully intact.
        await drop_pg_schema_if_exists(
            d["name"],
            s1,
            host=d["host"],
            port=d["port"],
            username=d["username"],
            password=d["password"],
        )
        assert not await _schema_exists(s1)
        assert await _schema_exists(s2)
        assert len(await a2.search("Entity_name", query_text="alice", limit=50)) == 1
    finally:
        await a1.close()
        await a2.close()


@pytest.mark.asyncio
async def test_postgres_graph_schema_isolation(two_schemas):
    """Two Postgres graph adapters pinned to different schemas stay isolated."""
    from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter

    d = _db()
    s1, s2 = two_schemas
    for s in (s1, s2):
        await create_pg_schema_if_not_exists(
            d["name"],
            s,
            host=d["host"],
            port=d["port"],
            username=d["username"],
            password=d["password"],
        )

    a1 = PostgresAdapter(connection_string=_base_url(), schema=s1)
    a2 = PostgresAdapter(connection_string=_base_url(), schema=s2)
    try:
        await a1.initialize()
        await a2.initialize()

        assert set(await _tables_in_schema(s1)) >= {"graph_node", "graph_edge"}
        assert set(await _tables_in_schema(s2)) >= {"graph_node", "graph_edge"}

        assert await a1.is_empty()
        await a1.add_node("n1", properties={"name": "Alice", "type": "Person"})
        await a1.add_node("n2", properties={"name": "Bob", "type": "Person"})
        await a1.add_edge("n1", "n2", "KNOWS", {"since": 2024})
        await a2.add_node("z1", properties={"name": "Zoe", "type": "Person"})

        nodes1, edges1 = await a1.get_graph_data()
        nodes2, edges2 = await a2.get_graph_data()
        assert len(nodes1) == 2 and len(edges1) == 1
        assert len(nodes2) == 1 and len(edges2) == 0
        assert not await a2.is_empty()

        # Dropping schema 1 leaves schema 2 intact.
        await drop_pg_schema_if_exists(
            d["name"],
            s1,
            host=d["host"],
            port=d["port"],
            username=d["username"],
            password=d["password"],
        )
        assert not await _schema_exists(s1)
        nodes2b, _ = await a2.get_graph_data()
        assert len(nodes2b) == 1
    finally:
        await a1.close()
        await a2.close()


@pytest.mark.asyncio
async def test_shared_handlers_create_and_delete_lifecycle():
    """End-to-end handler lifecycle: create_dataset provisions a schema, delete drops it.

    Requires cognee to be configured with a Postgres relational backend, since
    the shared handlers anchor to the relational configuration.
    """
    from cognee.infrastructure.databases.relational import get_relational_config

    if get_relational_config().db_provider != "postgres":
        pytest.skip("shared handler lifecycle requires DB_PROVIDER=postgres")
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable")

    from cognee.infrastructure.databases.vector.pgvector.PGVectorSharedDatasetDatabaseHandler import (
        PGVectorSharedDatasetDatabaseHandler as VH,
    )
    from cognee.infrastructure.databases.graph.postgres.PostgresGraphSharedDatasetDatabaseHandler import (
        PostgresGraphSharedDatasetDatabaseHandler as GH,
    )
    from cognee.modules.users.models import DatasetDatabase

    dataset_id = uuid.uuid4()
    schema = dataset_schema_name(dataset_id)

    vinfo = await VH.create_dataset(dataset_id, None)
    ginfo = await GH.create_dataset(dataset_id, None)
    try:
        assert vinfo["vector_dataset_database_handler"] == "pgvector_shared"
        assert ginfo["graph_dataset_database_handler"] == "postgres_graph_shared"
        assert vinfo["vector_database_connection_info"]["schema"] == schema
        assert ginfo["graph_database_connection_info"]["graph_database_schema"] == schema
        assert await _schema_exists(schema)
        assert set(await _tables_in_schema(schema)) >= {"graph_node", "graph_edge"}
    finally:
        record = DatasetDatabase(owner_id=uuid.uuid4(), dataset_id=dataset_id, **ginfo, **vinfo)
        await VH.delete_dataset(record)
        await GH.delete_dataset(record)

    assert not await _schema_exists(schema)
