"""cognify_status returns parseable JSON text and a structured payload.

The tool used to hand back ``str(status)`` — a Python repr an agent cannot
json-parse — with background errors pasted on as free text. These tests pin
the ToolResult contract: same payload as structured content and as JSON text.
"""

import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from src import server

DATASET_ID = "5f0c1b2a-3d4e-4f5a-9b6c-7d8e9f0a1b2c"


def _install_client(list_datasets=None, pipeline_status=None):
    server.cognee_client = SimpleNamespace(
        use_api=True,
        list_datasets=AsyncMock(
            return_value=list_datasets
            if list_datasets is not None
            else [{"id": DATASET_ID, "name": "main_dataset"}]
        ),
        get_pipeline_status=AsyncMock(return_value=pipeline_status or {}),
    )


@pytest.fixture(autouse=True)
def _restore_client():
    yield
    server.cognee_client = None


async def _call(args):
    async with Client(server.mcp) as client:
        return await client.call_tool("cognify_status", args)


async def test_single_pipeline_json_text_and_structured_payload():
    _install_client(pipeline_status={DATASET_ID: "completed"})
    result = await _call({"dataset_name": "main_dataset"})

    payload = {
        "dataset_name": "main_dataset",
        "pipelines": {DATASET_ID: "completed"},
        "background_errors": [],
    }
    assert result.structured_content == payload
    assert json.loads(result.content[0].text) == payload


async def test_multiple_pipelines_and_background_errors():
    server._task_errors["main_dataset"] = deque(
        [
            ("2026-11-03T09:30:00", "ingestion of notes.txt partially failed"),
            ("2026-11-03T10:00:00", "LLMConnectionError: provider timeout"),
        ]
    )

    # The tool queries each pipeline separately — the status endpoint answers
    # {dataset_id: status} per call — and nests the results itself.
    per_pipeline = {
        "cognify_pipeline": "completed",
        "add_pipeline": "running",
    }

    async def fake_get_status(dataset_ids, pipeline_name):
        return {DATASET_ID: per_pipeline[pipeline_name]}

    try:
        _install_client()
        server.cognee_client.get_pipeline_status = AsyncMock(side_effect=fake_get_status)
        result = await _call(
            {
                "dataset_name": "main_dataset",
                "pipelines": ["cognify_pipeline", "add_pipeline"],
            }
        )

        # Newest first, structured instead of free-form text.
        assert result.structured_content["background_errors"] == [
            {"time": "2026-11-03T10:00:00", "error": "LLMConnectionError: provider timeout"},
            {"time": "2026-11-03T09:30:00", "error": "ingestion of notes.txt partially failed"},
        ]
        assert result.structured_content["pipelines"] == {
            DATASET_ID: {"cognify_pipeline": "completed", "add_pipeline": "running"}
        }
        assert json.loads(result.content[0].text) == result.structured_content
    finally:
        server._task_errors.pop("main_dataset", None)


async def test_unknown_dataset_reports_error_text():
    _install_client(list_datasets=[])
    result = await _call({"dataset_name": "missing_dataset"})

    assert result.content[0].text.startswith("Error: Dataset 'missing_dataset' not found")


async def test_status_failure_uses_shared_error_prefix():
    failing = SimpleNamespace(
        use_api=True,
        list_datasets=AsyncMock(return_value=[{"id": DATASET_ID, "name": "main_dataset"}]),
        get_pipeline_status=AsyncMock(side_effect=RuntimeError("connection refused")),
    )
    server.cognee_client = failing
    result = await _call({"dataset_name": "main_dataset"})

    # Same "Cognify status failed: ..." shape the other tools use, so agents
    # can match one error pattern across tools.
    assert result.content[0].text.startswith("Cognify status failed: connection refused")
