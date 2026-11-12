"""An incremental update retires the conflict decisions an edit took the chosen fact from."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.update import incremental


def _patch_run(monkeypatch, bundle, write):
    order = []

    async def record_write(*args, **kwargs):
        order.append("write")
        return await write(*args, **kwargs)

    async def retire(dataset, user):
        order.append(("retire", dataset, user))
        return set()

    async def record_in_ledger(context, content_hash):
        order.append(("ledger", content_hash))

    monkeypatch.setattr(incremental, "_stage_and_plan", AsyncMock(return_value=bundle))
    monkeypatch.setattr(incremental, "record_data_item_content", record_in_ledger)
    monkeypatch.setattr(incremental, "generate_pipeline_id", lambda *args: uuid4())
    monkeypatch.setattr(
        incremental,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(incremental, "log_pipeline_run_complete", AsyncMock())
    monkeypatch.setattr(incremental, "log_pipeline_run_error", AsyncMock())
    monkeypatch.setattr(incremental, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(incremental, "_write_and_publish", record_write)
    monkeypatch.setattr(incremental, "retire_decisions_after_edit", retire)
    return order


CHANGED = {
    "status": "changed",
    "data_item": SimpleNamespace(content_hash="old-hash"),
    "staged": SimpleNamespace(content_hash="new-hash"),
}


async def _run(dataset, user):
    return await incremental._run_incremental_update(
        uuid4(), "new text", dataset, user, None, None, None, None, None, None, None, "llm"
    )


@pytest.mark.asyncio
async def test_decisions_are_retired_once_the_edit_is_written(monkeypatch):
    dataset, user = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    order = _patch_run(monkeypatch, CHANGED, AsyncMock(return_value={}))

    await _run(dataset, user)

    # The ledger is told the new content hash once the write has published it.
    assert order == ["write", ("ledger", "new-hash"), ("retire", dataset, user)]


@pytest.mark.asyncio
async def test_a_failed_write_retires_nothing(monkeypatch):
    dataset, user = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    order = _patch_run(monkeypatch, CHANGED, AsyncMock(side_effect=RuntimeError("write failed")))

    with pytest.raises(RuntimeError, match="write failed"):
        await _run(dataset, user)

    assert order == ["write"]
