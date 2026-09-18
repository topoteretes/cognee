"""Regression tests for the blast radius of ``PGVectorAdapter.prune()`` (#4956).

``prune()`` used to delegate unconditionally to the inherited
``SQLAlchemyAdapter.delete_database()``, which reflects and drops *every* table
in the schema. When Postgres is both the relational and the vector backend the
adapter borrows the relational engine (``_owns_engine=False``), so
``prune_system(vector=True, metadata=False)`` wiped ``users``, ``datasets``,
``alembic_version`` and the rest of the application schema despite the
documented ``metadata=False`` guarantee.

On that path ownership comes from the marker ``create_collection`` stamps as a table
comment, not from column shape — an app table holding an id, a JSON payload and an
embedding is identical to a collection. The tests pin both directions: a registered
collection is dropped, an exact-shape stranger survives.

These tests run against a real Postgres (default: cognee:cognee@localhost:5432)
and skip when it is unreachable, matching ``test_update_payload.py``.
"""

import socket
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from sqlalchemy import text

PG_HOST = "localhost"
PG_PORT = 5432


class StubEmbedder:
    """Minimal embedding engine — prune never embeds, so values are irrelevant."""

    dimensions = 4

    async def embed_text(self, texts):
        return [[0.0] * self.dimensions for _ in texts]

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_dimensions(self) -> int:
        return self.dimensions

    @property
    def model(self) -> str:
        return "stub"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


requires_postgres = pytest.mark.skipif(
    not _port_open(PG_HOST, PG_PORT), reason=f"no local Postgres on {PG_PORT}"
)


@asynccontextmanager
async def _throwaway_database():
    """Yield the URL of a fresh database with the pgvector extension enabled."""
    asyncpg = pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")

    database = f"prune_scope_{uuid4().hex[:8]}"
    admin = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user="cognee", password="cognee", database="postgres"
    )
    await admin.execute(f"CREATE DATABASE {database}")
    await admin.close()

    setup = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user="cognee", password="cognee", database=database
    )
    await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await setup.close()

    try:
        yield f"postgresql+asyncpg://cognee:cognee@{PG_HOST}:{PG_PORT}/{database}"
    finally:
        admin = await asyncpg.connect(
            host=PG_HOST, port=PG_PORT, user="cognee", password="cognee", database="postgres"
        )
        await admin.execute(f"DROP DATABASE {database} (FORCE)")
        await admin.close()


async def _table_names(engine) -> set:
    async with engine.begin() as connection:
        rows = await connection.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()")
        )
        return {row[0] for row in rows.all()}


async def _seed_relational_tables(engine) -> None:
    """Stand-ins for the application tables prune must not touch."""
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE users (id uuid PRIMARY KEY, email text)"))
        await connection.execute(text("CREATE TABLE datasets (id uuid PRIMARY KEY, name text)"))
        await connection.execute(
            text("CREATE TABLE alembic_version (version_num varchar(32) NOT NULL)")
        )
        await connection.execute(text("INSERT INTO alembic_version VALUES ('deadbeefcafe')"))


@asynccontextmanager
async def _borrowed_engine_adapter(monkeypatch, db_url: str):
    """Build a PGVectorAdapter that borrows a relational Postgres engine.

    Points the relational engine at the same database so the adapter's own
    ``__init__`` takes the shared-engine branch — the branch selection is part
    of what is under test, so it is never forced after the fact.
    """
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.pgvector import PGVectorAdapter as adapter_module
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    database = db_url.rsplit("/", 1)[1]
    relational = create_relational_engine(
        db_path="",
        db_name=database,
        db_host=PG_HOST,
        db_port=str(PG_PORT),
        db_username="cognee",
        db_password="cognee",
        db_provider="postgres",
    )
    monkeypatch.setattr(adapter_module, "get_relational_engine", lambda: relational)

    adapter = PGVectorAdapter(
        connection_string=db_url, api_key=None, embedding_engine=StubEmbedder()
    )
    assert adapter._owns_engine is False, "expected the borrowed-engine branch"
    assert adapter.engine is relational.engine

    try:
        yield adapter, relational
    finally:
        await relational.engine.dispose()
        create_relational_engine.cache_clear()


@pytest.mark.asyncio
@requires_postgres
async def test_prune_borrowed_engine_preserves_relational_schema(monkeypatch):
    """The reported bug: prune() must drop collections, not the application schema."""
    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            await _seed_relational_tables(relational.engine)
            await adapter.create_collection("Entity_name")
            await adapter.create_collection("DocumentChunk_text")

            before = await _table_names(relational.engine)
            assert {"Entity_name", "DocumentChunk_text", "users", "datasets"} <= before

            await adapter.prune()

            after = await _table_names(relational.engine)
            assert "Entity_name" not in after
            assert "DocumentChunk_text" not in after
            assert "users" in after
            assert "datasets" in after


@pytest.mark.asyncio
@requires_postgres
async def test_prune_borrowed_engine_preserves_alembic_version(monkeypatch):
    """Losing alembic_version corrupts migrations, so pin it separately."""
    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            await _seed_relational_tables(relational.engine)
            await adapter.create_collection("Entity_name")

            await adapter.prune()

            assert "alembic_version" in await _table_names(relational.engine)
            async with relational.engine.begin() as connection:
                revision = (
                    await connection.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar()
            assert revision == "deadbeefcafe", "the recorded revision must survive prune"


@pytest.mark.asyncio
@requires_postgres
async def test_prune_owned_engine_still_drops_whole_database(monkeypatch):
    """The owned-engine path is unchanged: that database is dedicated to vectors."""
    from cognee.infrastructure.databases.vector.pgvector import PGVectorAdapter as adapter_module
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    async with _throwaway_database() as db_url:
        # A sqlite relational engine forces __init__ down the own-engine branch.
        class _SqliteStub:
            db_uri = "sqlite+aiosqlite:///:memory:"

            class engine:
                class dialect:
                    name = "sqlite"

        monkeypatch.setattr(adapter_module, "get_relational_engine", lambda: _SqliteStub())

        adapter = PGVectorAdapter(
            connection_string=db_url, api_key=None, embedding_engine=StubEmbedder()
        )
        assert adapter._owns_engine is True, "expected the own-engine branch"

        try:
            await _seed_relational_tables(adapter.engine)
            await adapter.create_collection("Entity_name")

            await adapter.prune()

            assert await _table_names(adapter.engine) == set()
        finally:
            await adapter.engine.dispose()


@pytest.mark.asyncio
@requires_postgres
async def test_prune_borrowed_engine_ignores_vector_shaped_relational_tables(monkeypatch):
    """Near misses: tables that do not even reach the collection column shape."""
    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            async with relational.engine.begin() as connection:
                # Same column names, no pgvector column type.
                await connection.execute(
                    text("CREATE TABLE lookalike (id uuid PRIMARY KEY, payload json, vector text)")
                )
                # pgvector column, but not a collection (no payload column).
                await connection.execute(
                    text("CREATE TABLE embeddings_sidecar (id uuid PRIMARY KEY, vector vector(4))")
                )
            await adapter.create_collection("Entity_name")

            await adapter.prune()

            after = await _table_names(relational.engine)
            assert "Entity_name" not in after
            assert "lookalike" in after
            assert "embeddings_sidecar" in after


@pytest.mark.asyncio
@requires_postgres
async def test_prune_borrowed_engine_spares_exact_shape_application_table(monkeypatch):
    """Column shape is not proof of ownership.

    ``user_embeddings`` is everything a collection is, plus a column of its
    own — what a real app table storing embeddings looks like. Shape-based
    classification returned it alongside the genuine collection. Only the
    registered collection may go.
    """
    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            async with relational.engine.begin() as connection:
                await connection.execute(
                    text(
                        "CREATE TABLE user_embeddings ("
                        "  id uuid PRIMARY KEY,"
                        "  payload json,"
                        "  vector vector(4),"
                        "  created_at timestamptz DEFAULT now()"
                        ")"
                    )
                )
                # The same shape with no extra column: an exact structural twin
                # of a collection table, and still not one.
                await connection.execute(
                    text(
                        "CREATE TABLE app_embeddings "
                        "(id uuid PRIMARY KEY, payload json, vector vector(4))"
                    )
                )
                await connection.execute(
                    text("INSERT INTO user_embeddings (id) VALUES (gen_random_uuid())")
                )
            await adapter.create_collection("Entity_name")

            await adapter.prune()

            after = await _table_names(relational.engine)
            assert "Entity_name" not in after, "the registered collection must still be dropped"
            assert "user_embeddings" in after
            assert "app_embeddings" in after
            async with relational.engine.begin() as connection:
                surviving_rows = (
                    await connection.execute(text("SELECT count(*) FROM user_embeddings"))
                ).scalar()
            assert surviving_rows == 1, "the unrelated table's rows must survive intact"


@pytest.mark.asyncio
@requires_postgres
async def test_prune_borrowed_engine_ignores_marker_on_non_collection_table(monkeypatch):
    """The marker alone does not authorize a drop.

    A comment can be copied onto another table (``INCLUDING COMMENTS``, or by
    hand), so ownership needs the marker *and* the collection shape.
    """
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        _COLLECTION_OWNERSHIP_MARKER,
    )

    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            async with relational.engine.begin() as connection:
                await connection.execute(
                    text("CREATE TABLE audit_log (id uuid PRIMARY KEY, message text)")
                )
                await connection.execute(
                    text(f"COMMENT ON TABLE audit_log IS '{_COLLECTION_OWNERSHIP_MARKER}'")
                )

            await adapter.prune()

            assert "audit_log" in await _table_names(relational.engine)


@pytest.mark.asyncio
@requires_postgres
async def test_prune_borrowed_engine_leaves_unmarked_collection_in_place(monkeypatch):
    """Unproven ownership fails closed, even at the cost of an incomplete prune.

    A collection written by a cognee older than the marker carries no proof, so
    prune leaves it rather than guess.
    """
    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            await adapter.create_collection("Entity_name")
            async with relational.engine.begin() as connection:
                await connection.execute(text('COMMENT ON TABLE "Entity_name" IS NULL'))

            async with adapter.engine.begin() as connection:
                _, owned, unproven = await adapter._classify_schema_tables(connection)
            assert owned == []
            assert unproven == ["Entity_name"]

            await adapter.prune()

            assert "Entity_name" in await _table_names(relational.engine)


@pytest.mark.asyncio
@requires_postgres
async def test_create_collection_backfills_marker_on_pre_marker_table(monkeypatch):
    """Reopening a legacy collection claims it, so upgrades do not leak forever.

    Failing closed would otherwise strand every collection created before the
    marker existed.
    """
    async with _throwaway_database() as db_url:
        async with _borrowed_engine_adapter(monkeypatch, db_url) as (adapter, relational):
            # A collection exactly as a pre-marker cognee left it: right shape,
            # right name, no ownership comment.
            async with relational.engine.begin() as connection:
                await connection.execute(
                    text(
                        'CREATE TABLE "Entity_name" '
                        "(id uuid PRIMARY KEY, payload json, vector vector(4))"
                    )
                )
                await connection.execute(
                    text(
                        "CREATE TABLE user_embeddings (id uuid PRIMARY KEY, payload json, vector vector(4))"
                    )
                )

            await adapter.create_collection("Entity_name")

            await adapter.prune()

            after = await _table_names(relational.engine)
            assert "Entity_name" not in after, "the reopened collection must be claimed and dropped"
            assert "user_embeddings" in after, (
                "the backfill must not claim tables cognee never opened"
            )
