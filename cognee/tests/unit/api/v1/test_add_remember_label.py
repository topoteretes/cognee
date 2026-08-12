"""The add and remember endpoints accept a `label` form field for uploads.

A provided label wraps each upload in a DataItem before it reaches the
add/remember pipeline; ingestion unwraps the DataItem and stores the label
on the item's Data record. Without a label the uploads pass through
unchanged. On remember, a label is rejected when combined with session_id
or content_type — those storage paths never create Data records, so the
label would be silently dropped.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from cognee.api.client import app
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.ingestion.data_item import DataItem

add_pkg = importlib.import_module("cognee.api.v1.add")
remember_pkg = importlib.import_module("cognee.api.v1.remember")


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="test@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


def pipeline_run_completed():
    return PipelineRunCompleted(
        pipeline_run_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        dataset_name="test_dataset",
    )


UPLOADS = [
    ("data", ("first.txt", b"first file", "text/plain")),
    ("data", ("second.txt", b"second file", "text/plain")),
]


def test_add_with_label_wraps_uploads_in_data_items(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "label": "quarterly-report"},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.label for item in sent] == ["quarterly-report", "quarterly-report"]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


def test_add_without_label_passes_uploads_through(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS[:1],
            # Swagger UI submits untouched optional fields as "".
            data={"datasetName": "test_dataset", "label": ""},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert not any(isinstance(item, DataItem) for item in sent)
        assert sent[0].filename == "first.txt"


def test_remember_with_label_wraps_uploads_in_data_items(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "label": "quarterly-report"},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.label for item in sent] == ["quarterly-report", "quarterly-report"]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


def test_remember_without_label_passes_uploads_through(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", "label": ""},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert not any(isinstance(item, DataItem) for item in sent)


@pytest.mark.parametrize(
    "extra_form",
    [{"session_id": "claude-code-1718000000"}, {"content_type": "skills"}],
    ids=["session_id", "content_type"],
)
def test_remember_rejects_label_outside_normal_ingestion(client, extra_form):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", "label": "quarterly-report", **extra_form},
        )

        assert response.status_code == 400
        assert "label is only supported for normal ingestion" in response.json()["detail"]
        mock_remember.assert_not_awaited()
