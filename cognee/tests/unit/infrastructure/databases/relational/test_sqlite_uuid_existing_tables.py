"""UUID columns of an existing SQLite database become text on upgrade.

Tables created before the ``sqlite_uuid`` compile hook declare ``UUID``, which
gives them NUMERIC affinity: a hex UUID that SQLite reads as a number (all
digits, or digits around a single ``e``) is stored as a REAL, and reading it
back fails in SQLAlchemy's UUID result processor.
"""

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
import sqlalchemy as sa

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.migrations import lockstep
from cognee.modules.migrations.startup import MigrationError, run_relational_migrations
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence

NUMERIC_LOOKING_UUID = UUID("12345678901234567890123456789012")
# SQLite reads this one as 1234567e123456789012345678901234, which is infinity.
EXPONENT_LOOKING_UUID = UUID("1234567e123456789012345678901234")

# A revision that predates this fix; upgrading from it runs every later one.
STAMPED_REVISION = "e3f1a9c5b7d2"

# DESC is not in the shipped index; it shows the rebuild keeps index SQL verbatim.
EVIDENCE_INDEX = (
    "CREATE INDEX ix_prov_evidence_edge ON provenance_edge_evidence (dataset_id, edge_id DESC)"
)

# The table as migration f3a7b9c1d2e4 built it before the compile hook.
PRE_HOOK_DDL = (
    """
    CREATE TABLE provenance_edge_evidence (
        id UUID NOT NULL,
        tenant_id UUID,
        user_id UUID NOT NULL,
        dataset_id UUID NOT NULL,
        data_id UUID NOT NULL,
        pipeline_run_id UUID,
        chunk_id UUID NOT NULL,
        chunk_index INTEGER,
        edge_id UUID NOT NULL,
        source_node_id UUID NOT NULL,
        destination_node_id UUID NOT NULL,
        relationship_name TEXT NOT NULL,
        evidence_kind VARCHAR(32) DEFAULT 'extracted' NOT NULL,
        source_task VARCHAR(255),
        confidence FLOAT,
        created_at DATETIME NOT NULL,
        PRIMARY KEY (id)
    )
    """,
    EVIDENCE_INDEX,
    "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)",
    f"INSERT INTO alembic_version VALUES ('{STAMPED_REVISION}')",
)


def _evidence(edge_id: UUID) -> ProvenanceEdgeEvidence:
    return ProvenanceEdgeEvidence(
        user_id=uuid4(),
        dataset_id=uuid4(),
        data_id=uuid4(),
        chunk_id=uuid4(),
        edge_id=edge_id,
        source_node_id=uuid4(),
        destination_node_id=uuid4(),
        relationship_name="contains",
    )


async def _pre_hook_database(tmp_path, *statements: str):
    """A pre-hook database holding one row whose edge id is already a REAL."""
    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path}/cognee_db")
    kept = _evidence(uuid4())
    async with adapter.engine.begin() as connection:
        for statement in PRE_HOOK_DDL + statements:
            await connection.execute(sa.text(statement))
    async with adapter.get_async_session() as session:
        session.add_all([kept, _evidence(NUMERIC_LOOKING_UUID)])
        await session.commit()
    return adapter, kept.id


async def _upgrade(adapter) -> None:
    await run_relational_migrations("head", lockstep.packaged_script_location(), engine=adapter)


@pytest_asyncio.fixture
async def upgraded_pre_hook_database(tmp_path):
    adapter, kept_id = await _pre_hook_database(tmp_path)
    await _upgrade(adapter)
    yield adapter, kept_id
    await adapter.engine.dispose()


async def _schema(adapter) -> list[tuple]:
    async with adapter.engine.connect() as connection:
        rows = await connection.execute(sa.text("SELECT name, sql FROM sqlite_master"))
        return rows.all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edge_id", [NUMERIC_LOOKING_UUID, EXPONENT_LOOKING_UUID], ids=["digits", "exponent"]
)
async def test_numeric_looking_uuid_round_trips_after_upgrade(upgraded_pre_hook_database, edge_id):
    adapter, _ = upgraded_pre_hook_database
    written = _evidence(edge_id)
    async with adapter.get_async_session() as session:
        session.add(written)
        await session.commit()

    async with adapter.get_async_session() as session:
        stored = await session.scalar(
            sa.select(ProvenanceEdgeEvidence.edge_id).where(ProvenanceEdgeEvidence.id == written.id)
        )

    assert stored == edge_id


@pytest.mark.asyncio
async def test_upgrade_drops_evidence_whose_ids_were_already_numbers(
    upgraded_pre_hook_database,
):
    adapter, kept_id = upgraded_pre_hook_database
    async with adapter.get_async_session() as session:
        ids = (await session.scalars(sa.select(ProvenanceEdgeEvidence.id))).all()

    assert ids == [kept_id]


@pytest.mark.asyncio
async def test_upgrade_refuses_numbers_in_other_tables_and_changes_nothing(tmp_path):
    adapter, _ = await _pre_hook_database(
        tmp_path,
        "CREATE TABLE records (id UUID NOT NULL PRIMARY KEY, owner_id UUID)",
        f"INSERT INTO records VALUES ('{uuid4().hex}', '{NUMERIC_LOOKING_UUID.hex}')",
    )
    before = await _schema(adapter)

    with pytest.raises(MigrationError) as raised:
        await _upgrade(adapter)

    assert "records (owner_id), 1 row(s)" in str(raised.value.__cause__)
    assert await _schema(adapter) == before
    async with adapter.engine.connect() as connection:
        evidence_rows = await connection.scalar(
            sa.text("SELECT count(*) FROM provenance_edge_evidence")
        )
        version = await connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
    assert (evidence_rows, version) == (2, STAMPED_REVISION)
    await adapter.engine.dispose()


@pytest.mark.asyncio
async def test_upgrade_keeps_the_table_indexes(upgraded_pre_hook_database):
    adapter, _ = upgraded_pre_hook_database
    async with adapter.engine.connect() as connection:
        index_sql = await connection.scalar(
            sa.text("SELECT sql FROM sqlite_master WHERE name = 'ix_prov_evidence_edge'")
        )

    assert index_sql == EVIDENCE_INDEX


@pytest.mark.asyncio
async def test_upgrade_interrupted_before_the_rename_keeps_the_table(tmp_path):
    adapter, _ = await _pre_hook_database(tmp_path)
    before = await _schema(adapter)

    def interrupt_rename(conn, cursor, statement, *args):
        if statement.startswith('ALTER TABLE "_uuid_as_text_'):
            raise RuntimeError("interrupted")

    # Migrations open their own engine, so listen on every engine.
    sa.event.listen(sa.engine.Engine, "before_cursor_execute", interrupt_rename)
    try:
        with pytest.raises(MigrationError):
            await _upgrade(adapter)
    finally:
        sa.event.remove(sa.engine.Engine, "before_cursor_execute", interrupt_rename)

    assert await _schema(adapter) == before
    async with adapter.engine.connect() as connection:
        evidence_rows = await connection.scalar(
            sa.text("SELECT count(*) FROM provenance_edge_evidence")
        )
    assert evidence_rows == 2

    await _upgrade(adapter)
    async with adapter.engine.connect() as connection:
        evidence_rows = await connection.scalar(
            sa.text("SELECT count(*) FROM provenance_edge_evidence")
        )
        index_sql = await connection.scalar(
            sa.text("SELECT sql FROM sqlite_master WHERE name = 'ix_prov_evidence_edge'")
        )
    assert (evidence_rows, index_sql) == (1, EVIDENCE_INDEX)
    await adapter.engine.dispose()
