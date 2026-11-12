"""add() and remember() take the grouped ``dlt_config`` and still the bare DLT kwargs.

``POST /api/v1/add`` takes it as one JSON form field.
"""

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("dlt")

from cognee.api.client import app
from cognee.api.v1.add.add import add
from cognee.api.v1.remember.remember import _ADD_ONLY, RememberKwargs
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.ingestion.dlt_config import DLT_OPTION_NAMES


def test_add_takes_dlt_config():
    assert "dlt_config" in inspect.signature(add).parameters


def test_remember_forwards_dlt_config_to_add():
    assert "dlt_config" in _ADD_ONLY
    assert "dlt_config" in RememberKwargs.__annotations__


def test_every_add_only_kwarg_is_an_add_parameter_or_a_dlt_option():
    """remember() forwards these blindly; add() must know each one."""
    accepted = set(inspect.signature(add).parameters) | DLT_OPTION_NAMES
    assert _ADD_ONLY <= accepted, _ADD_ONLY - accepted


def _post_add(dlt_config: str):
    """POST /api/v1/add with one CSV and a ``dlt_config`` form field; add() is mocked."""
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=str(uuid4()))
    try:
        with (
            patch.object(
                importlib.import_module("cognee.api.v1.add"), "add", new_callable=AsyncMock
            ) as mock_add,
            TestClient(app) as client,
        ):
            mock_add.return_value = PipelineRunCompleted(
                pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="tables"
            )
            response = client.post(
                "/api/v1/add",
                files=[("data", ("rows.csv", b"id,name\n1,a\n", "text/csv"))],
                data={"datasetName": "tables", "dlt_config": dlt_config},
            )
    finally:
        app.dependency_overrides.pop(get_authenticated_user, None)
    return response, mock_add


def test_add_endpoint_forwards_dlt_config():
    response, mock_add = _post_add('{"primary_key": "id", "write_disposition": "replace"}')

    assert response.status_code == 200
    assert mock_add.call_args.kwargs["dlt_config"] == {
        "primary_key": "id",
        "write_disposition": "replace",
    }


def test_add_endpoint_without_dlt_config_sends_none():
    response, mock_add = _post_add("")

    assert response.status_code == 200
    assert mock_add.call_args.kwargs["dlt_config"] is None


@pytest.mark.parametrize(
    "dlt_config",
    ["primary_key=id", '["id"]', '{"primary_keys": "id"}', '{"max_rows_per_table": "many"}'],
    ids=["not_json", "not_an_object", "unknown_key", "wrong_type"],
)
def test_add_endpoint_rejects_invalid_dlt_config(dlt_config):
    response, mock_add = _post_add(dlt_config)

    assert response.status_code == 400
    assert "dlt_config" in response.json()["detail"]
    mock_add.assert_not_awaited()
