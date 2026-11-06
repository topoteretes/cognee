"""``POST /v1/add`` and ``POST /v1/cognify`` carry ``codegraph_config`` like ``/remember``.

The routes are pure here: ``add()`` / ``cognify()`` are monkeypatched on their
packages (the routers import them lazily), so no databases are touched. The same
validation applies on every surface: both spellings, a 400 naming the problem for
a mistyped option, and nothing forwarded when the field is empty.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.cognify.routers.get_cognify_router import get_cognify_router
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user

add_pkg = importlib.import_module("cognee.api.v1.add")
cognify_pkg = importlib.import_module("cognee.api.v1.cognify")

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())
REPO_URL = "https://github.com/org/repo"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/api/v1/add")
    app.include_router(get_cognify_router(), prefix="/api/v1/cognify")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


def _completed():
    return PipelineRunCompleted(
        pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="test_dataset"
    )


@pytest.mark.parametrize("field", ["codegraph_config", "codegraphConfig"])
def test_add_forwards_the_config_under_either_spelling(client, field):
    config = {"treat_as_repository": True, "include_documents": False, "repo_credentials": "tok"}
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = _completed()

        response = client.post(
            "/api/v1/add",
            data={"datasetName": "test_dataset", "raw_data": [REPO_URL], field: json.dumps(config)},
        )

        assert response.status_code == 200, response.text
        assert mock_add.call_args.kwargs["codegraph_config"] == config


def test_add_sends_no_config_for_plain_ingestion(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = _completed()

        response = client.post(
            "/api/v1/add", data={"datasetName": "test_dataset", "raw_data": ["a note"]}
        )

        assert response.status_code == 200, response.text
        assert mock_add.call_args.kwargs["codegraph_config"] is None


@pytest.mark.parametrize(
    "value, message",
    [
        ("{not json", "not valid JSON"),
        ('{"index_vector": true}', "Unknown codegraph_config keys: index_vector"),
        ('{"treat_as_repository": "yes"}', "must be a bool"),
    ],
)
def test_add_rejects_an_invalid_config_with_a_400(client, value, message):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            data={"datasetName": "test_dataset", "raw_data": [REPO_URL], "codegraph_config": value},
        )

        assert response.status_code == 400, response.text
        assert message in response.json()["detail"]
        mock_add.assert_not_awaited()


@pytest.mark.parametrize("field", ["codegraph_config", "codegraphConfig"])
def test_cognify_forwards_the_config_under_either_spelling(client, field):
    with patch.object(cognify_pkg, "cognify", new_callable=AsyncMock) as mock_cognify:
        mock_cognify.return_value = {}

        response = client.post(
            "/api/v1/cognify", json={"datasets": ["test_dataset"], field: {"index_vectors": True}}
        )

        assert response.status_code == 200, response.text
        assert mock_cognify.call_args.kwargs["codegraph_config"] == {"index_vectors": True}


def test_cognify_sends_no_config_when_omitted(client):
    with patch.object(cognify_pkg, "cognify", new_callable=AsyncMock) as mock_cognify:
        mock_cognify.return_value = {}

        response = client.post("/api/v1/cognify", json={"datasets": ["test_dataset"]})

        assert response.status_code == 200, response.text
        assert mock_cognify.call_args.kwargs["codegraph_config"] is None


@pytest.mark.parametrize(
    "value, message",
    [
        ({"index_vector": True}, "Unknown codegraph_config keys: index_vector"),
        ({"index_vectors": "true"}, "must be a bool"),
    ],
)
def test_cognify_rejects_an_invalid_config_with_a_400(client, value, message):
    with patch.object(cognify_pkg, "cognify", new_callable=AsyncMock) as mock_cognify:
        response = client.post(
            "/api/v1/cognify", json={"datasets": ["test_dataset"], "codegraph_config": value}
        )

        assert response.status_code == 400, response.text
        assert message in response.json()["detail"]
        mock_cognify.assert_not_awaited()
