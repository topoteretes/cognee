"""data at the cognify() boundary."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.data.models import Data

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


def _row():
    return Data(id=uuid4(), name="row")


@pytest.mark.asyncio
@pytest.mark.parametrize("datasets", [None, ["a", "b"]])
async def test_data_requires_exactly_one_dataset(datasets):
    with pytest.raises(ValueError, match="exactly one dataset"):
        await cognify_module.cognify(datasets=datasets, data=[_row()])


@pytest.mark.asyncio
async def test_empty_data_is_rejected():
    with pytest.raises(ValueError, match="at least one Data row"):
        await cognify_module.cognify(datasets="a", data=[])


@pytest.mark.asyncio
async def test_data_is_rejected_for_dry_run_and_remote(monkeypatch):
    with pytest.raises(ValueError, match="does not accept data"):
        await cognify_module.cognify(datasets="a", data=[_row()], dry_run=True)

    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())
    with pytest.raises(ValueError, match="remote Cognee instance"):
        await cognify_module.cognify(datasets="a", data=[_row()])


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[])
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
async def test_data_is_passed_to_the_pipeline(
    mock_get_resolver,
    mock_migrations,
    mock_get_default_tasks,
    mock_executor,
    mock_remote,
):
    executor = AsyncMock(return_value={})
    mock_executor.return_value = executor
    dataset_id = uuid4()
    rows = [_row()]

    await cognify_module.cognify(datasets=dataset_id, data=rows, extractor="llm")

    assert executor.await_args.kwargs["data"] is rows
    assert executor.await_args.kwargs["datasets"] == [dataset_id]
    assert "data" not in mock_get_default_tasks.await_args.kwargs
