"""The add and remember endpoints accept per-file `label` form entries.

Labels pair positionally with the uploaded files: the Nth label applies to
the Nth file, and an empty entry skips that file. When any label is given,
each upload is wrapped in a DataItem before it reaches the add/remember
pipeline; ingestion unwraps the DataItem and stores the label on the file's
Data record. Without labels the uploads pass through unchanged. A label
count that doesn't match the file count is rejected, as is a label combined
with session_id or content_type on remember — those storage paths never
create Data records, so the label would be silently dropped.
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


def test_add_pairs_each_label_with_its_upload(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "label": ["finance", "engineering"]},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.label for item in sent] == ["finance", "engineering"]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


def test_add_empty_label_entry_skips_that_file(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "label": ["finance", ""]},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [item.label for item in sent] == ["finance", None]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


@pytest.mark.parametrize(
    "label_form",
    # Swagger UI submits untouched array entries as ""; plain clients omit
    # the field entirely. Both must behave as "no labels".
    [{"label": ["", ""]}, {}],
    ids=["empty_entries", "field_omitted"],
)
def test_add_without_labels_passes_uploads_through(client, label_form):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", **label_form},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert not any(isinstance(item, DataItem) for item in sent)
        assert sent[0].filename == "first.txt"


@pytest.mark.parametrize(
    "files,labels",
    [
        (UPLOADS, ["finance"]),
        (UPLOADS, ["a", "b", "c"]),
        # An empty entry still counts toward the pairing — it means "no label
        # for that file", not "one fewer label".
        (UPLOADS, ["finance", "people", ""]),
        (None, ["orphan"]),
    ],
    ids=["fewer_labels", "more_labels", "more_labels_with_empty", "labels_without_files"],
)
def test_add_rejects_label_count_mismatch(client, files, labels):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=files,
            data={"datasetName": "test_dataset", "label": labels},
        )

        assert response.status_code == 400
        assert "one label per uploaded file" in response.json()["detail"]
        mock_add.assert_not_awaited()


def test_remember_pairs_each_label_with_its_upload(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "label": ["finance", "engineering"]},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.label for item in sent] == ["finance", "engineering"]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


@pytest.mark.parametrize(
    "label_form",
    [{"label": [""]}, {}],
    ids=["empty_entries", "field_omitted"],
)
def test_remember_without_labels_passes_uploads_through(client, label_form):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", **label_form},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert not any(isinstance(item, DataItem) for item in sent)


@pytest.mark.parametrize(
    "files,labels",
    [
        (UPLOADS, ["finance"]),
        (UPLOADS, ["a", "b", "c"]),
        (None, ["orphan"]),
    ],
    ids=["fewer_labels", "more_labels", "labels_without_files"],
)
def test_remember_rejects_label_count_mismatch(client, files, labels):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=files,
            data={"datasetName": "test_dataset", "label": labels},
        )

        assert response.status_code == 400
        assert "one label per uploaded file" in response.json()["detail"]
        mock_remember.assert_not_awaited()


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
            data={"datasetName": "test_dataset", "label": ["finance"], **extra_form},
        )

        assert response.status_code == 400
        assert "label is only supported for normal ingestion" in response.json()["detail"]
        mock_remember.assert_not_awaited()
