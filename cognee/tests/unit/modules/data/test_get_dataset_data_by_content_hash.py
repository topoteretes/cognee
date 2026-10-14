"""Content-hash lookup of Data rows, against an isolated in-memory database.

``Data.content_hash`` has always been the dedup key, served by the
``data_dataset_content_lookup`` index — but nothing exposed a lookup by it, so
callers that needed "the row holding this content" listed the dataset and
compared hashes themselves. These tests pin the public lookup and the
``compute_content_hash`` helper it is keyed on.
"""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.data.models import Data
from cognee.modules.ingestion.data_types.TextData import TextData


def test_compute_content_hash_matches_what_ingestion_records():
    """The helper IS the identity formula: equal to TextData's identifier."""
    text = "Vasilije prefers uv over poetry for Python projects"
    assert compute_content_hash(text) == TextData(text).get_identifier()
    assert compute_content_hash(text) == compute_content_hash(text.encode("utf-8"))
    assert len(compute_content_hash(text)) == 32  # MD5 hex digest


@pytest.mark.asyncio
async def test_lookup_matches_either_hash_column_scoped_to_dataset(monkeypatch):
    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data_by_content_hash")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )

    dataset_id, other_dataset_id = uuid4(), uuid4()
    owner_a, owner_b = uuid4(), uuid4()
    wanted = compute_content_hash("the content we are looking for")
    now = datetime.now(timezone.utc)

    older = Data(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_a,
        content_hash=wanted,
        created_at=now - timedelta(minutes=5),
    )
    newer_by_raw_hash = Data(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_b,
        content_hash="unrelated",
        raw_content_hash=wanted,
        created_at=now,
    )
    unrelated = Data(id=uuid4(), dataset_id=dataset_id, owner_id=owner_a, content_hash="x")
    same_hash_elsewhere = Data(
        id=uuid4(), dataset_id=other_dataset_id, owner_id=owner_a, content_hash=wanted
    )

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Data.__table__.create)
        async with sessions() as session:
            session.add_all([older, newer_by_raw_hash, unrelated, same_hash_elsewhere])
            await session.commit()

        rows = await module.get_dataset_data_by_content_hash(dataset_id, wanted)
        # Both hash columns match; newest first; the other dataset is excluded.
        assert [row.id for row in rows] == [newer_by_raw_hash.id, older.id]

        scoped = await module.get_dataset_data_by_content_hash(dataset_id, wanted, owner_id=owner_a)
        assert [row.id for row in scoped] == [older.id]

        assert await module.get_dataset_data_by_content_hash(dataset_id, "missing") == []
        assert await module.get_dataset_data_by_content_hash(uuid4(), wanted) == []
    finally:
        await engine.dispose()
