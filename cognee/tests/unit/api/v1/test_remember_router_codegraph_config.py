"""``POST /v1/remember`` carries ``codegraph_config`` and ignores the removed code type.

All tests are pure: ``remember()`` itself is monkeypatched on the
``cognee.api.v1.remember`` package (the router imports it lazily at call
time), so no databases, git, enola, or network are involved. The tests cover
the router-level contract: the JSON field (both spellings), its validation,
the deprecated ``index_vectors`` and ``content_type='code'`` inputs, and the
paths that refuse the options.
"""

import importlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.users.methods import get_authenticated_user

remember_package = importlib.import_module("cognee.api.v1.remember")

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())
REPO_URL = "https://github.com/org/repo"


class FakeRememberResult:
    """Minimal stand-in for RememberResult, matching the .to_dict() contract."""

    def __init__(self, status="running", dataset_name="code_ds"):
        self.status = status
        self.dataset_name = dataset_name

    def to_dict(self):
        return {
            "status": self.status,
            "dataset_name": self.dataset_name,
            "dataset_id": str(uuid4()),
            "items_processed": 0,
        }


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


@pytest.fixture
def fake_remember(monkeypatch):
    """Capture the args the router forwards to remember()."""
    captured = {}

    async def _fake(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return FakeRememberResult()

    monkeypatch.setattr(remember_package, "remember", _fake)
    return captured


@pytest.mark.parametrize("field", ["codegraph_config", "codegraphConfig"])
def test_the_config_is_forwarded_under_either_spelling(client, fake_remember, field):
    config = {"treat_as_repository": True, "include_documents": False, "repo_credentials": "tok"}

    response = client.post(
        "/remember",
        data={"datasetName": "code_ds", "raw_data": [REPO_URL], field: json.dumps(config)},
    )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == [REPO_URL]
    assert fake_remember["kwargs"]["codegraph_config"] == config
    assert fake_remember["kwargs"]["content_type"] is None


def test_plain_ingestion_sends_no_config(client, fake_remember):
    response = client.post("/remember", data={"datasetName": "ds", "raw_data": ["a note"]})

    assert response.status_code == 200, response.text
    assert "codegraph_config" not in fake_remember["kwargs"]
    assert "index_vectors" not in fake_remember["kwargs"]


def test_the_two_spellings_must_agree(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "ds",
            "raw_data": [REPO_URL],
            "codegraph_config": json.dumps({"index_vectors": True}),
            "codegraphConfig": json.dumps({"index_vectors": False}),
        },
    )

    assert response.status_code == 400
    assert "same field" in response.json()["detail"]
    assert not fake_remember


@pytest.mark.parametrize(
    "value, message",
    [
        ("{not json", "not valid JSON"),
        ('["index_vectors"]', "must be a JSON object"),
        ('{"index_vector": true}', "Unknown codegraph_config keys: index_vector"),
        ('{"include_documents": "false"}', "must be a bool"),
        ('{"repo_credentials": ""}', "non-empty str"),
    ],
)
def test_an_invalid_config_is_a_400_naming_the_problem(client, fake_remember, value, message):
    response = client.post(
        "/remember",
        data={"datasetName": "ds", "raw_data": [REPO_URL], "codegraph_config": value},
    )

    assert response.status_code == 400, response.text
    assert message in response.json()["detail"]
    assert not fake_remember


def test_a_blank_config_is_no_config(client, fake_remember):
    # Swagger UI submits an untouched field as "".
    response = client.post(
        "/remember", data={"datasetName": "ds", "raw_data": ["a note"], "codegraph_config": ""}
    )

    assert response.status_code == 200, response.text
    assert "codegraph_config" not in fake_remember["kwargs"]


def test_deprecated_index_vectors_folds_into_the_config(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "raw_data": [REPO_URL],
            "index_vectors": "true",
            "codegraph_config": json.dumps({"include_documents": False}),
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["kwargs"]["codegraph_config"] == {
        "index_vectors": True,
        "include_documents": False,
    }


@pytest.mark.parametrize(
    "extra", [{"session_id": "s1"}, {"content_type": "skills", "skills_text": "# skill"}]
)
@pytest.mark.parametrize(
    "option", [{"index_vectors": "true"}, {"codegraph_config": '{"treat_as_repository": true}'}]
)
def test_options_are_refused_where_add_and_cognify_never_run(client, fake_remember, extra, option):
    data = {"datasetName": "ds", **extra, **option}
    if "session_id" in extra:
        data["raw_data"] = ["a note"]

    response = client.post("/remember", data=data)

    assert response.status_code == 400, response.text
    assert "codegraph_config" in response.json()["detail"]
    assert not fake_remember


def test_a_config_that_restates_the_defaults_is_fine_with_a_session(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "ds",
            "raw_data": ["a note"],
            "session_id": "s1",
            "index_vectors": "false",
            "codegraph_config": json.dumps({"include_documents": True}),
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["kwargs"]["session_id"] == "s1"


def test_the_removed_code_content_type_is_ignored(client, fake_remember, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/remember",
            data={
                "datasetName": "code_ds",
                "content_type": "code",
                "raw_data": [REPO_URL, "", "  "],
                "index_vectors": "true",
                "run_in_background": "true",
            },
        )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == [REPO_URL]
    assert fake_remember["kwargs"]["content_type"] is None
    assert fake_remember["kwargs"]["codegraph_config"] == {"index_vectors": True}
    assert fake_remember["kwargs"]["run_in_background"] is True
    assert "content_type='code' is deprecated" in caplog.text


def test_the_removed_code_content_type_still_rejects_uploads(client, fake_remember):
    # The old request contract stays: the payload is raw_data repository specs.
    response = client.post(
        "/remember",
        data={"datasetName": "ds", "content_type": "code", "raw_data": [REPO_URL]},
        files=[("data", ("main.py", b"print('hi')", "text/plain"))],
    )

    assert response.status_code == 400
    assert "does not accept file uploads" in response.json()["detail"]
    assert not fake_remember


@pytest.mark.parametrize(
    "form",
    [
        {"content_type": "code"},
        {"codegraph_config": '{"treat_as_repository": true}'},
    ],
)
def test_server_local_repository_paths_are_a_400_when_local_paths_are_disabled(
    client, fake_remember, monkeypatch, form
):
    storage_module = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")
    monkeypatch.setattr(storage_module.settings, "accept_local_file_path", False)

    response = client.post(
        "/remember", data={"datasetName": "ds", "raw_data": ["/srv/some/repo"], **form}
    )

    assert response.status_code == 400
    assert "ACCEPT_LOCAL_FILE_PATH" in response.json()["detail"]
    assert not fake_remember

    # A git URL is unaffected: nothing is read from the server's filesystem.
    response = client.post("/remember", data={"datasetName": "ds", "raw_data": [REPO_URL], **form})
    assert response.status_code == 200, response.text


def test_unknown_content_type_still_rejected(client, fake_remember):
    response = client.post("/remember", data={"datasetName": "ds", "content_type": "sculpture"})

    assert response.status_code == 400
    assert "Unsupported content_type" in response.json()["detail"]
    assert not fake_remember
