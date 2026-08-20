"""identify_data(): identify()'s lookup, returning the row instead of its id.

Contract under test:
  - same hash + dataset + owner + tenant → the Data row identify() points at
  - miss (other owner / other dataset / unknown hash) → None
  - an explicitly passed session is used as-is (no second session opened)
  - identify_data and identify agree on the winning row
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data
from cognee.modules.ingestion.identify import identify, identify_data

_ENGINE_PATCH = "cognee.modules.ingestion.identify.get_relational_engine"


async def _make_engine(rows: list[dict]) -> tuple[SQLAlchemyAdapter, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp.name}")
    await engine.create_database()
    async with engine.get_async_session() as session:
        for row in rows:
            session.add(Data(**row))
        await session.commit()
    return engine, tmp.name


def _user(tenant_id=None):
    return SimpleNamespace(id=uuid4(), tenant_id=tenant_id)


def _row(*, dataset_id, owner_id, content_hash, tenant_id=None, pipeline_status=None):
    return dict(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_id,
        tenant_id=tenant_id,
        name="doc.txt",
        content_hash=content_hash,
        raw_data_location="file:///tmp/doc.txt",
        pipeline_status=pipeline_status or {},
        token_count=-1,
    )


def _classified(content_hash: str):
    async def aget_identifier():
        return content_hash

    return SimpleNamespace(get_identifier=lambda: content_hash, aget_identifier=aget_identifier)


@pytest.mark.asyncio
async def test_hit_returns_the_row_identify_points_at():
    user = _user()
    dataset_id = uuid4()
    status = {"add_pipeline": {str(dataset_id): "DATA_ITEM_PROCESSING_COMPLETED"}}
    seeded = _row(
        dataset_id=dataset_id, owner_id=user.id, content_hash="h1", pipeline_status=status
    )
    engine, db_path = await _make_engine([seeded])
    try:
        with patch(_ENGINE_PATCH, return_value=engine):
            row = await identify_data(_classified("h1"), user, dataset_id)
            same_id = await identify(_classified("h1"), user, dataset_id)
        assert row is not None
        assert row.id == seeded["id"] == same_id
        # The whole point: the row carries the columns the caller needs next.
        assert row.pipeline_status == status
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_miss_scopes_like_identify():
    user = _user()
    stranger = _user()
    dataset_id = uuid4()
    engine, db_path = await _make_engine(
        [_row(dataset_id=dataset_id, owner_id=user.id, content_hash="h1")]
    )
    try:
        with patch(_ENGINE_PATCH, return_value=engine):
            assert await identify_data(_classified("h1"), stranger, dataset_id) is None
            assert await identify_data(_classified("h1"), user, uuid4()) is None
            assert await identify_data(_classified("unknown"), user, dataset_id) is None
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_explicit_session_is_used_without_opening_another():
    """The incremental wrapper resolves fresh content inside the session that
    then records its pipeline status — one connection, not two."""
    user = _user()
    dataset_id = uuid4()
    seeded = _row(dataset_id=dataset_id, owner_id=user.id, content_hash="h1")
    engine, db_path = await _make_engine([seeded])
    try:
        # Any attempt to open a session through the module's engine accessor
        # would be a regression: the caller's session must be the only one.
        with patch(_ENGINE_PATCH, side_effect=AssertionError("must not open a session")):
            async with engine.get_async_session() as session:
                row = await identify_data(_classified("h1"), user, dataset_id, session=session)
                assert row is not None and row.id == seeded["id"]
                # The row is attached to the caller's session, so it can be
                # updated and committed right there.
                row.pipeline_status = {"add_pipeline": {str(dataset_id): "done"}}
                await session.commit()
        async with engine.get_async_session() as session:
            refreshed = await session.get(Data, seeded["id"])
            assert refreshed.pipeline_status == {"add_pipeline": {str(dataset_id): "done"}}
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
