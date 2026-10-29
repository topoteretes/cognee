"""MCP forget: the reply wording follows the receipt's ``status``.

``cognee.forget(data_id=...)`` returns the delete receipt, whose ``status`` is
``"success"`` or ``"not_found"`` (nothing existed for the id: a wrong id, or an
already-deleted item). The tool must never describe the second case as a
deletion.
"""

import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

DATA_ID = "11111111-1111-1111-1111-111111111111"
DATASET_ID = "22222222-2222-2222-2222-222222222222"


class RecordingForgetClient:
    def __init__(self, result):
        self.calls: list[dict] = []
        self._result = result

    async def forget(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


def _receipt(status):
    return {
        "status": status,
        "dataset_id": DATASET_ID,
        "data_id": DATA_ID,
        "data_record_found": status == "success",
        "deleted_nodes": 3 if status == "success" else 0,
        "deleted_edges": 1 if status == "success" else 0,
        "data_remaining": False if status == "success" else None,
        "dataset_deleted": False,
    }


@pytest.mark.asyncio
async def test_forget_data_item_success_reads_as_deleted(monkeypatch):
    from src import server

    monkeypatch.setattr(server, "cognee_client", RecordingForgetClient(_receipt("success")))

    [reply] = await server.forget(dataset="ds", data_id=DATA_ID)

    assert reply.text == f"Data item '{DATA_ID}' deleted (status=success)."


@pytest.mark.asyncio
async def test_forget_data_item_not_found_never_reads_as_deleted(monkeypatch):
    from src import server

    fake_client = RecordingForgetClient(_receipt("not_found"))
    monkeypatch.setattr(server, "cognee_client", fake_client)

    [reply] = await server.forget(dataset="ds", data_id=DATA_ID)

    assert "not found" in reply.text
    assert "nothing was deleted" in reply.text
    assert "status=not_found" in reply.text
    assert "' deleted" not in reply.text
    assert str(fake_client.calls[0]["data_id"]) == DATA_ID


@pytest.mark.asyncio
async def test_forget_dataset_wording_is_unchanged(monkeypatch):
    from src import server

    monkeypatch.setattr(
        server,
        "cognee_client",
        RecordingForgetClient({"dataset_id": DATASET_ID, "status": "success"}),
    )

    [reply] = await server.forget(dataset="ds")

    assert reply.text == "Dataset 'ds' deleted (status=success)."
