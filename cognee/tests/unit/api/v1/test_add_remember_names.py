"""The add and remember endpoints accept a `names` form field, like `labels`.

`names` names raw_data text, which is otherwise stored as `text_<md5>`. It uses
the same wire format as `labels` (a JSON array, or the comma-separated form
Swagger UI sends) and pairs the same way: the Nth entry applies to the Nth item,
uploads first, then raw_data entries, and an empty entry skips that item. A
named item is wrapped in a DataItem carrying the name. An upload is named by
its own filename, so a name for one is rejected, as are a count mismatch,
non-string JSON entries, and names on remember's session or content_type paths.
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


UPLOAD = [("data", ("first.txt", b"first file", "text/plain"))]


def test_add_names_each_raw_data_text(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            data={
                "datasetName": "test_dataset",
                "raw_data": ["alpha text", "beta text"],
                "names": '["Q3 plan, final", "Notes"]',
            },
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.data for item in sent] == ["alpha text", "beta text"]
        assert [item.name for item in sent] == ["Q3 plan, final", "Notes"]


def test_add_pairs_names_after_uploads_with_an_empty_entry_for_the_upload(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOAD,
            data={"datasetName": "test_dataset", "raw_data": ["alpha text"], "names": ",Notes"},
        )

        assert response.status_code == 200
        upload, text = mock_add.call_args.args[0]
        assert upload.name is None and upload.data.filename == "first.txt"
        assert (text.data, text.name) == ("alpha text", "Notes")


def test_add_pairs_names_and_labels_together(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            data={
                "datasetName": "test_dataset",
                "raw_data": ["alpha text"],
                "labels": '["finance"]',
                "names": '["Q3 plan"]',
            },
        )

        assert response.status_code == 200
        (item,) = mock_add.call_args.args[0]
        assert (item.label, item.name) == ("finance", "Q3 plan")


@pytest.mark.parametrize(
    "names_form",
    [{"names": ""}, {}, {"names": '["", " "]'}],
    ids=["field_blank", "field_omitted", "empty_entries"],
)
def test_add_without_names_passes_text_through(client, names_form):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            data={"datasetName": "test_dataset", "raw_data": ["a", "b"], **names_form},
        )

        assert response.status_code == 200
        assert mock_add.call_args.args[0] == ["a", "b"]


@pytest.mark.parametrize(
    "form,expected",
    [
        ({"raw_data": ["a", "b"], "names": '["one"]'}, "one name per data item"),
        ({"raw_data": ["a"], "names": '["one", "two"]'}, "one name per data item"),
        ({"raw_data": ["a"], "names": "[1]"}, "names must be a JSON array of strings"),
    ],
    ids=["fewer_names", "more_names", "non_string_entry"],
)
def test_add_rejects_malformed_names(client, form, expected):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post("/api/v1/add", data={"datasetName": "test_dataset", **form})

        assert response.status_code == 400
        assert expected in response.json()["detail"]
        mock_add.assert_not_awaited()


def test_add_rejects_a_name_for_an_upload(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=UPLOAD,
            data={"datasetName": "test_dataset", "names": '["Renamed"]'},
        )

        assert response.status_code == 400
        assert "an uploaded file is named by its filename" in response.json()["detail"]
        mock_add.assert_not_awaited()


def test_remember_names_raw_data_text(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            data={"datasetName": "test_dataset", "raw_data": ["alpha text"], "names": '["Q3"]'},
        )

        assert response.status_code == 200
        (item,) = mock_remember.call_args.args[0]
        assert (item.data, item.name) == ("alpha text", "Q3")


@pytest.mark.parametrize(
    "extra_form",
    [{"session_id": "s1"}, {"content_type": "skills"}],
    ids=["session_id", "content_type"],
)
def test_remember_rejects_names_outside_normal_ingestion(client, extra_form):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=UPLOAD,
            data={"datasetName": "test_dataset", "names": '["Q3"]', **extra_form},
        )

        assert response.status_code == 400
        assert "only supported for normal ingestion" in response.json()["detail"]
        mock_remember.assert_not_awaited()
