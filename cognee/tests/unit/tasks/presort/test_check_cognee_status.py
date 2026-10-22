import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.tasks.presort.check_cognee_status import check_cognee_status
from cognee.tasks.presort.models import FileRecord

MODULE = "cognee.tasks.presort.check_cognee_status"
ENGINE_TARGET = "cognee.infrastructure.databases.relational.get_relational_engine"


def _records():
    return [
        FileRecord(path="/f/new.txt", name="new.txt", content_hash="hash_new"),
        FileRecord(path="/f/staged.txt", name="staged.txt", content_hash="hash_staged"),
        FileRecord(path="/f/done.txt", name="done.txt", content_hash="hash_done"),
        FileRecord(path="/f/unhashed.bin", name="unhashed.bin"),  # no hash -> unknown
    ]


def _fake_engine(rows):
    result = MagicMock()
    result.fetchall.return_value = rows
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def get_async_session():
        yield session

    engine = MagicMock()
    engine.get_async_session = get_async_session
    return engine


def _fake_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.tenant_id = None
    return user


@pytest.mark.asyncio
async def test_statuses_filled():
    dataset_id = uuid.uuid4()
    rows = [
        # (content_hash, pipeline_status, dataset_id, dataset_name)
        ("hash_staged", {}, dataset_id, "inbox"),
        (
            "hash_done",
            {"cognify_pipeline": {str(dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
            dataset_id,
            "archive",
        ),
    ]
    records = _records()

    with patch(ENGINE_TARGET, return_value=_fake_engine(rows)):
        warnings = await check_cognee_status(records, user=_fake_user())

    assert warnings == []
    by_name = {record.name: record for record in records}
    assert by_name["new.txt"].cognee_status == "new"
    assert by_name["staged.txt"].cognee_status == "staged"
    assert by_name["staged.txt"].known_in_datasets == ["inbox"]
    assert by_name["done.txt"].cognee_status == "cognified"
    assert by_name["done.txt"].known_in_datasets == ["archive"]
    assert by_name["unhashed.bin"].cognee_status == "unknown"


@pytest.mark.asyncio
async def test_db_failure_leaves_unknown():
    records = _records()
    with patch(ENGINE_TARGET, side_effect=RuntimeError("no database")):
        warnings = await check_cognee_status(records, user=_fake_user())

    assert len(warnings) == 1
    assert "unknown" in warnings[0]
    assert all(record.cognee_status == "unknown" for record in records)


@pytest.mark.asyncio
async def test_no_hashes_short_circuits():
    record = FileRecord(path="/f/unhashed.bin", name="unhashed.bin")
    warnings = await check_cognee_status([record], user=_fake_user())
    assert warnings == []
    assert record.cognee_status == "unknown"
