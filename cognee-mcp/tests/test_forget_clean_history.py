"""The MCP ``forget`` tool forwards ``clean_history`` and reports a history-only call."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from src import server


async def test_clean_history_alone_is_forwarded_and_reported(monkeypatch):
    fake = SimpleNamespace(
        forget=AsyncMock(return_value={"history_cleaned": True, "status": "success"})
    )
    monkeypatch.setattr(server, "cognee_client", fake)

    out = await server.forget(clean_history=True)

    assert fake.forget.call_args.kwargs["clean_history"] is True
    assert "Search history deleted" in out[0].text


async def test_clean_history_with_an_item_is_forwarded(monkeypatch):
    fake = SimpleNamespace(
        forget=AsyncMock(return_value={"status": "success", "history_cleaned": True})
    )
    monkeypatch.setattr(server, "cognee_client", fake)
    data_id, dataset_id = uuid4(), uuid4()

    out = await server.forget(data_id=str(data_id), dataset_id=str(dataset_id), clean_history=True)

    kwargs = fake.forget.call_args.kwargs
    assert kwargs["clean_history"] is True
    assert kwargs["data_id"] == data_id and kwargs["dataset_id"] == dataset_id
    assert "Data item" in out[0].text


async def test_forget_defaults_to_keeping_history(monkeypatch):
    fake = SimpleNamespace(forget=AsyncMock(return_value={"status": "success"}))
    monkeypatch.setattr(server, "cognee_client", fake)

    await server.forget(dataset="docs")

    assert fake.forget.call_args.kwargs["clean_history"] is False


async def test_no_target_and_no_clean_history_is_an_error_without_a_call(monkeypatch):
    fake = SimpleNamespace(forget=AsyncMock())
    monkeypatch.setattr(server, "cognee_client", fake)

    out = await server.forget()

    fake.forget.assert_not_awaited()
    assert out[0].text.startswith("Error:") and "clean_history" in out[0].text
