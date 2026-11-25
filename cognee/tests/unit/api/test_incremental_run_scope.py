"""An incremental update's work runs under its own run id, so its LLM calls can be found by it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.update import incremental
from cognee.modules.operations.usage_accumulator import get_parent_run_id


def _patch_run(monkeypatch, bundle):
    seen = {}

    async def write(*args, **kwargs):
        seen["write"] = get_parent_run_id()
        return {}

    async def repair(*args, **kwargs):
        seen["repair"] = get_parent_run_id()
        return {}

    monkeypatch.setattr(incremental, "_stage_and_plan", AsyncMock(return_value=bundle))
    monkeypatch.setattr(incremental, "_remove_unreferenced_files", AsyncMock())
    monkeypatch.setattr(incremental, "record_data_item_content", AsyncMock())
    monkeypatch.setattr(incremental, "_record_content_in_ledger", AsyncMock())
    monkeypatch.setattr(incremental, "generate_pipeline_id", lambda *args: uuid4())
    monkeypatch.setattr(
        incremental,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(incremental, "log_pipeline_run_complete", AsyncMock())
    monkeypatch.setattr(incremental, "log_pipeline_run_error", AsyncMock())
    monkeypatch.setattr(incremental, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(incremental, "_write_and_publish", write)
    monkeypatch.setattr(incremental, "_repair_unchanged", repair)
    monkeypatch.setattr(incremental, "retire_decisions_after_edit", AsyncMock(return_value=set()))
    return seen


async def _run():
    dataset, user = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    return await incremental._run_incremental_update(
        uuid4(), "new text", dataset, user, None, None, None, None, None, None, None, "llm"
    )


@pytest.mark.asyncio
async def test_an_edit_runs_under_the_run_id_update_returns(monkeypatch):
    seen = _patch_run(
        monkeypatch,
        {
            "status": "changed",
            "data_item": SimpleNamespace(content_hash="old-hash"),
            "staged": SimpleNamespace(content_hash="new-hash"),
        },
    )

    result = await _run()

    assert seen["write"] == result["pipeline_run_id"]
    assert get_parent_run_id() is None


@pytest.mark.asyncio
async def test_a_repair_runs_under_the_run_id_update_returns(monkeypatch):
    seen = _patch_run(
        monkeypatch,
        {
            "status": "unchanged",
            "repairs": ["missing chunk"],
            "stored_count": 1,
            "data_item": SimpleNamespace(content_hash="same-hash"),
            "staged": SimpleNamespace(content_hash="same-hash"),
        },
    )

    result = await _run()

    assert seen["repair"] == result["pipeline_run_id"]
    assert get_parent_run_id() is None
