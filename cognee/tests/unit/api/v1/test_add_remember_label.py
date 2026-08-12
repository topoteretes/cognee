"""The add and remember endpoints accept `labels` and `external_metadata` form fields.

Each field is a single JSON-encoded array (e.g. '["finance", ""]' or
'[{"source": "crm"}, null]') because multipart clients — Swagger UI included —
cannot reliably repeat array form fields (swagger-api/swagger-ui#10221
serializes repeated entries into one comma-joined part). Entries pair
positionally: the Nth entry applies to the Nth file, and an empty entry
("" / null / {}) skips that file. When any entry is given, each upload is
wrapped in a DataItem before it reaches the add/remember pipeline; ingestion
unwraps the DataItem and stores the attributes on the file's Data record.
Without either field the uploads pass through unchanged. Malformed JSON, a
count that doesn't match the file count, and either field combined with
session_id or content_type on remember are all rejected with 400.
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
            data={"datasetName": "test_dataset", "labels": '["finance", "engineering"]'},
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
            data={"datasetName": "test_dataset", "labels": '["finance", ""]'},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [item.label for item in sent] == ["finance", None]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


@pytest.mark.parametrize(
    "labels_form",
    # Swagger UI submits an untouched field as ""; plain clients omit it; an
    # all-empty array carries no label. All must behave as "no labels".
    [{"labels": ""}, {}, {"labels": '["", ""]'}],
    ids=["field_blank", "field_omitted", "empty_entries"],
)
def test_add_without_labels_passes_uploads_through(client, labels_form):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", **labels_form},
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert not any(isinstance(item, DataItem) for item in sent)
        assert sent[0].filename == "first.txt"


@pytest.mark.parametrize(
    "files,labels",
    [
        (UPLOADS, '["finance"]'),
        (UPLOADS, '["a", "b", "c"]'),
        # An empty entry still counts toward the pairing — it means "no label
        # for that file", not "one fewer label".
        (UPLOADS, '["finance", "people", ""]'),
        (None, '["orphan"]'),
    ],
    ids=["fewer_labels", "more_labels", "more_labels_with_empty", "labels_without_files"],
)
def test_add_rejects_label_count_mismatch(client, files, labels):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=files,
            data={"datasetName": "test_dataset", "labels": labels},
        )

        assert response.status_code == 400
        assert "one label per uploaded file" in response.json()["detail"]
        mock_add.assert_not_awaited()


@pytest.mark.parametrize(
    "labels,expected",
    [
        ("finance", "not valid JSON"),
        ('"finance"', "JSON array of strings"),
        ("[1, 2]", "JSON array of strings"),
    ],
    ids=["bare_string", "json_string_not_array", "non_string_entries"],
)
def test_add_rejects_malformed_labels(client, labels, expected):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "labels": labels},
        )

        assert response.status_code == 400
        assert expected in response.json()["detail"]
        mock_add.assert_not_awaited()


def test_remember_pairs_each_label_with_its_upload(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "labels": '["finance", "engineering"]'},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.label for item in sent] == ["finance", "engineering"]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


@pytest.mark.parametrize(
    "labels_form",
    [{"labels": ""}, {}],
    ids=["field_blank", "field_omitted"],
)
def test_remember_without_labels_passes_uploads_through(client, labels_form):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", **labels_form},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert not any(isinstance(item, DataItem) for item in sent)


@pytest.mark.parametrize(
    "files,labels",
    [
        (UPLOADS, '["finance"]'),
        (UPLOADS, '["a", "b", "c"]'),
        (None, '["orphan"]'),
    ],
    ids=["fewer_labels", "more_labels", "labels_without_files"],
)
def test_remember_rejects_label_count_mismatch(client, files, labels):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=files,
            data={"datasetName": "test_dataset", "labels": labels},
        )

        assert response.status_code == 400
        assert "one label per uploaded file" in response.json()["detail"]
        mock_remember.assert_not_awaited()


@pytest.mark.parametrize(
    "attribute_form",
    [{"labels": '["finance"]'}, {"external_metadata": '[{"source": "crm"}]'}],
    ids=["labels", "external_metadata"],
)
@pytest.mark.parametrize(
    "extra_form",
    [{"session_id": "claude-code-1718000000"}, {"content_type": "skills"}],
    ids=["session_id", "content_type"],
)
def test_remember_rejects_attributes_outside_normal_ingestion(client, extra_form, attribute_form):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", **attribute_form, **extra_form},
        )

        assert response.status_code == 400
        assert "only supported for normal ingestion" in response.json()["detail"]
        mock_remember.assert_not_awaited()


def test_add_pairs_each_metadata_entry_with_its_upload(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={
                "datasetName": "test_dataset",
                "external_metadata": '[{"source": "crm", "ticket": 42}, null]',
            },
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.external_metadata for item in sent] == [
            {"source": "crm", "ticket": 42},
            None,
        ]
        assert [item.label for item in sent] == [None, None]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


def test_add_pairs_labels_and_metadata_together(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={
                "datasetName": "test_dataset",
                "labels": '["finance", ""]',
                "external_metadata": '[{}, {"source": "wiki"}]',
            },
        )

        assert response.status_code == 200
        sent = mock_add.call_args.args[0]
        assert [item.label for item in sent] == ["finance", None]
        assert [item.external_metadata for item in sent] == [None, {"source": "wiki"}]


@pytest.mark.parametrize(
    "files,external_metadata",
    [
        (UPLOADS, '[{"source": "crm"}]'),
        (UPLOADS, '[{"a": 1}, {"b": 2}, {"c": 3}]'),
        (None, '[{"source": "crm"}]'),
    ],
    ids=["fewer_entries", "more_entries", "entries_without_files"],
)
def test_add_rejects_metadata_count_mismatch(client, files, external_metadata):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=files,
            data={"datasetName": "test_dataset", "external_metadata": external_metadata},
        )

        assert response.status_code == 400
        assert "one external_metadata entry per uploaded file" in response.json()["detail"]
        mock_add.assert_not_awaited()


@pytest.mark.parametrize(
    "external_metadata,expected",
    [
        ("source=crm", "not valid JSON"),
        ('{"source": "crm"}', "JSON array of objects"),
        ('["crm"]', "JSON array of objects"),
        ('[{"node_set": ["x"]}]', "reserved key 'node_set'"),
    ],
    ids=["bare_string", "object_not_array", "non_object_entries", "reserved_node_set"],
)
def test_add_rejects_malformed_metadata(client, external_metadata, expected):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", "external_metadata": external_metadata},
        )

        assert response.status_code == 400
        assert expected in response.json()["detail"]
        mock_add.assert_not_awaited()


def test_remember_pairs_metadata_with_its_upload(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS[:1],
            data={"datasetName": "test_dataset", "external_metadata": '[{"source": "crm"}]'},
        )

        assert response.status_code == 200
        sent = mock_remember.call_args.args[0]
        assert [item.external_metadata for item in sent] == [{"source": "crm"}]
