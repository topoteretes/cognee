"""The add and remember endpoints accept a ``data_ids`` form field.

Same wire convention as ``labels`` / ``external_metadata``: one JSON array,
paired positionally with the uploaded files, ``null`` to let the server mint
that file's id. A pinned id lands on the file's DataItem so ingestion stores
the document under it — the handle a remote SDK caller needs for a later
``PATCH /api/v1/update``. Malformed entries, a count mismatch, and pins
combined with session_id / content_type on remember are rejected with 400.
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
from cognee.tasks.ingestion.data_item import DataItem, pair_labels_with_data, parse_data_ids
from cognee.tasks.ingestion.exceptions import DataIdCountMismatchError, InvalidDataIdsError

add_pkg = importlib.import_module("cognee.api.v1.add")
remember_pkg = importlib.import_module("cognee.api.v1.remember")

UPLOADS = [
    ("data", ("first.txt", b"first", "text/plain")),
    ("data", ("second.txt", b"second", "text/plain")),
]


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
        pipeline_run_id=uuid.uuid4(), dataset_id=uuid.uuid4(), dataset_name="test_dataset"
    )


# ----- parsing / pairing -----


def test_parse_data_ids_accepts_uuids_and_null_entries():
    pinned = uuid.uuid4()
    assert parse_data_ids(f'["{pinned}", null, ""]') == [pinned, None, None]
    assert parse_data_ids(None) is None
    assert parse_data_ids("   ") is None


@pytest.mark.parametrize("raw", ['["not-a-uuid"]', "[1]", '{"a": 1}', "not json"])
def test_parse_data_ids_rejects_malformed_values(raw):
    with pytest.raises(InvalidDataIdsError):
        parse_data_ids(raw)


def test_pair_sets_data_id_on_each_item_and_checks_count():
    pinned = uuid.uuid4()
    paired = pair_labels_with_data(["a", "b"], None, None, [pinned, None])
    assert [type(item) for item in paired] == [DataItem, DataItem]
    assert [item.data_id for item in paired] == [pinned, None]
    assert [item.label for item in paired] == [None, None]

    assert pair_labels_with_data(["a"], None, None, [None]) == ["a"]
    with pytest.raises(DataIdCountMismatchError):
        pair_labels_with_data(["a", "b"], None, None, [pinned])


# ----- add route -----


def test_add_pins_each_id_to_its_upload(client):
    pinned = uuid.uuid4()
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "data_ids": f'["{pinned}", null]'},
        )

        assert response.status_code == 200, response.text
        sent = mock_add.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert [item.data_id for item in sent] == [pinned, None]
        assert [item.data.filename for item in sent] == ["first.txt", "second.txt"]


def test_add_pins_ids_alongside_labels(client):
    pinned = uuid.uuid4()
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={
                "datasetName": "test_dataset",
                "labels": '["finance", ""]',
                "data_ids": f'[null, "{pinned}"]',
            },
        )

        assert response.status_code == 200, response.text
        sent = mock_add.call_args.args[0]
        assert [item.label for item in sent] == ["finance", None]
        assert [item.data_id for item in sent] == [None, pinned]


@pytest.mark.parametrize(
    "data_ids",
    [f'["{uuid.uuid4()}"]', '["nope", null]', "[1, 2]"],
    ids=["count_mismatch", "not_a_uuid", "not_strings"],
)
def test_add_rejects_bad_data_ids(client, data_ids):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "data_ids": data_ids},
        )

        assert response.status_code == 400, response.text
        mock_add.assert_not_called()


# ----- remember route -----


def test_remember_pins_each_id_to_its_upload(client):
    pinned = uuid.uuid4()
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = SimpleNamespace(
            status="completed", to_dict=lambda: {"status": "completed"}
        )

        response = client.post(
            "/api/v1/remember",
            files=UPLOADS,
            data={"datasetName": "test_dataset", "data_ids": f'["{pinned}", null]'},
        )

        assert response.status_code == 200, response.text
        sent = mock_remember.call_args.args[0]
        assert [item.data_id for item in sent] == [pinned, None]


@pytest.mark.parametrize(
    "extra_form",
    [{"session_id": "s1"}, {"content_type": "skills"}],
    ids=["session_id", "content_type"],
)
def test_remember_rejects_pins_outside_normal_ingestion(client, extra_form):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=UPLOADS,
            data={
                "datasetName": "test_dataset",
                "data_ids": f'["{uuid.uuid4()}", null]',
                **extra_form,
            },
        )

        assert response.status_code == 400, response.text
        assert "data_ids" in response.json()["detail"]
        mock_remember.assert_not_called()


# ----- update route: dataset_name -----


def test_update_route_accepts_dataset_name_and_requires_exactly_one_selector(client):
    data_id = uuid.uuid4()
    update_pkg = importlib.import_module("cognee.api.v1.update")
    with patch.object(update_pkg, "update", new_callable=AsyncMock) as mock_update:
        mock_update.return_value = {
            "status": "unchanged",
            "regions": 0,
            "deleted_chunks": 0,
            "added_chunks": 0,
            "reused_chunks": 0,
            "kept_chunks": 0,
            "reindexed_chunks": 0,
        }

        response = client.patch(
            "/api/v1/update",
            params={"data_id": str(data_id), "dataset_name": "repro"},
            files=UPLOADS[:1],
        )
        assert response.status_code == 200, response.text
        assert mock_update.call_args.kwargs["dataset_name"] == "repro"
        assert mock_update.call_args.kwargs["dataset_id"] is None

        mock_update.reset_mock()
        response = client.patch(
            "/api/v1/update", params={"data_id": str(data_id)}, files=UPLOADS[:1]
        )
        assert response.status_code == 422, response.text
        response = client.patch(
            "/api/v1/update",
            params={"data_id": str(data_id), "dataset_id": str(uuid.uuid4()), "dataset_name": "x"},
            files=UPLOADS[:1],
        )
        assert response.status_code == 422, response.text
        mock_update.assert_not_called()
