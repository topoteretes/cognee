"""Content-hash lookup of Data rows, against an isolated in-memory database.

``Data.content_hash`` has always been the dedup key, served by the
``data_dataset_content_lookup`` index — but nothing exposed a lookup by it, so
callers that needed "the row holding this content" listed the dataset and
compared hashes themselves. These tests pin the public lookup and the
``compute_content_hash`` helper it is keyed on.
"""

import importlib
from datetime import datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.modules.data.content_hash import compute_content_hash, normalize_content_hash
from cognee.modules.data.models import Data
from cognee.modules.ingestion.data_types.TextData import TextData


def test_compute_content_hash_matches_what_ingestion_records():
    """The helper IS the identity formula: equal to TextData's identifier."""
    text = "Vasilije prefers uv over poetry for Python projects"
    assert compute_content_hash(text) == TextData(text).get_identifier()
    assert compute_content_hash(text) == compute_content_hash(text.encode("utf-8"))
    assert len(compute_content_hash(text)) == 32  # MD5 hex digest


@pytest.mark.asyncio
async def test_compute_content_hash_equals_the_streaming_file_hash():
    """Binary ingestion hashes files in chunks; the formula must be the same MD5."""
    payload = b"\x00binary payload\xff" * 5000
    assert compute_content_hash(payload) == await get_file_content_hash(BytesIO(payload))


def test_normalize_content_hash_lowercases_and_validates():
    digest = compute_content_hash("x")
    assert normalize_content_hash(digest) == digest
    assert normalize_content_hash(f" {digest.upper()} ") == digest
    for bad in ("", "abc", digest[:-1], digest + "0", "g" * 32):
        with pytest.raises(ValueError):
            normalize_content_hash(bad)


@pytest_asyncio.fixture
async def db(monkeypatch):
    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data_by_content_hash")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    async with engine.begin() as connection:
        await connection.run_sync(Data.__table__.create)
    try:
        yield SimpleNamespace(lookup=module.get_dataset_data_by_content_hash, sessions=sessions)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_lookup_is_scoped_like_ingestion_dedup(db):
    """Matches on content_hash only, for the caller's own rows in the dataset.

    The predicate is the one ingestion dedup uses — dataset, hash, owner,
    tenant — which is also the column order of ``data_dataset_content_lookup``.
    """
    dataset_id, other_dataset_id = uuid4(), uuid4()
    owner_a, owner_b, tenant = uuid4(), uuid4(), uuid4()
    me = SimpleNamespace(id=owner_a, tenant_id=None)
    wanted = compute_content_hash("the content we are looking for")
    now = datetime.now(timezone.utc)

    older = Data(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_a,
        content_hash=wanted,
        created_at=now - timedelta(minutes=5),
    )
    newer = Data(
        id=uuid4(), dataset_id=dataset_id, owner_id=owner_a, content_hash=wanted, created_at=now
    )
    # Same content added by another writer to the shared dataset: not mine.
    other_writers = Data(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_b,
        content_hash=wanted,
        created_at=now + timedelta(minutes=1),
    )
    # Mine, but in another tenant: ingestion dedup would not reuse it either.
    other_tenant = Data(
        id=uuid4(), dataset_id=dataset_id, owner_id=owner_a, tenant_id=tenant, content_hash=wanted
    )
    # A PDF whose loader output happens to equal the text: raw_content_hash is
    # derived content, so it is not a match.
    pdf_with_same_text = Data(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_a,
        content_hash="hash-of-the-pdf-bytes",
        raw_content_hash=wanted,
        created_at=now + timedelta(minutes=2),
    )
    unrelated = Data(id=uuid4(), dataset_id=dataset_id, owner_id=owner_a, content_hash="x")
    same_hash_elsewhere = Data(
        id=uuid4(), dataset_id=other_dataset_id, owner_id=owner_a, content_hash=wanted
    )

    async with db.sessions() as session:
        session.add_all(
            [
                older,
                newer,
                other_writers,
                other_tenant,
                pdf_with_same_text,
                unrelated,
                same_hash_elsewhere,
            ]
        )
        await session.commit()

    rows = await db.lookup(dataset_id, wanted, me)
    assert [row.id for row in rows] == [newer.id, older.id]

    # The other writer finds only their own row.
    assert [
        row.id
        for row in await db.lookup(dataset_id, wanted, SimpleNamespace(id=owner_b, tenant_id=None))
    ] == [other_writers.id]
    # Scoped to my tenant.
    assert [
        row.id
        for row in await db.lookup(
            dataset_id, wanted, SimpleNamespace(id=owner_a, tenant_id=tenant)
        )
    ] == [other_tenant.id]

    assert await db.lookup(dataset_id, "missing", me) == []
    assert await db.lookup(uuid4(), wanted, me) == []


@pytest.mark.asyncio
async def test_lookup_pages_like_the_listing(db):
    dataset_id, owner = uuid4(), uuid4()
    me = SimpleNamespace(id=owner, tenant_id=None)
    empty = compute_content_hash("")
    now = datetime.now(timezone.utc)
    rows = [
        Data(
            id=uuid4(),
            dataset_id=dataset_id,
            owner_id=owner,
            content_hash=empty,
            created_at=now - timedelta(seconds=i),
        )
        for i in range(5)
    ]
    async with db.sessions() as session:
        session.add_all(rows)
        await session.commit()

    newest_first = [row.id for row in rows]
    assert [r.id for r in await db.lookup(dataset_id, empty, me)] == newest_first
    assert [r.id for r in await db.lookup(dataset_id, empty, me, limit=2)] == newest_first[:2]
    assert [r.id for r in await db.lookup(dataset_id, empty, me, limit=2, offset=2)] == (
        newest_first[2:4]
    )
    assert await db.lookup(dataset_id, empty, me, limit=2, offset=5) == []
