"""Unit tests for the content_type='code' branch of the /v1/remember router.

All tests are pure: ``remember()`` itself is monkeypatched on the
``cognee.api.v1.remember`` package (the router imports it lazily at call
time), so no databases, git, enola, or network are involved. The tests cover
the router-level contract: validation, payload shaping, and passthrough of
the code-only options.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.users.methods import get_authenticated_user

remember_package = importlib.import_module("cognee.api.v1.remember")
storage_module = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())


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
def app():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return app


@pytest.fixture
def client(app):
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


def test_code_specs_are_forwarded(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "repositories": ["https://github.com/org/repo", "https://github.com/org/other"],
            "index_vectors": "true",
            "run_in_background": "true",
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == [
        "https://github.com/org/repo",
        "https://github.com/org/other",
    ]
    assert fake_remember["kwargs"]["content_type"] == "code"
    assert fake_remember["kwargs"]["index_vectors"] is True
    assert fake_remember["kwargs"]["run_in_background"] is True


def test_empty_repository_entries_are_dropped(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            # Swagger UI submits untouched array items as "".
            "repositories": ["https://github.com/org/repo", "", "  "],
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == ["https://github.com/org/repo"]
    # index_vectors defaults off and must not leak to non-code kwargs handling.
    assert fake_remember["kwargs"]["index_vectors"] is False


def test_code_requires_repositories(client, fake_remember):
    response = client.post(
        "/remember",
        data={"datasetName": "code_ds", "content_type": "code"},
    )

    assert response.status_code == 400
    assert "repositories" in response.json()["detail"]
    assert not fake_remember


def test_code_rejects_file_uploads(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "repositories": ["https://github.com/org/repo"],
        },
        files=[("data", ("main.py", b"print('hi')", "text/plain"))],
    )

    assert response.status_code == 400
    assert "file uploads" in response.json()["detail"]
    assert not fake_remember


def test_code_rejects_session_id(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "repositories": ["https://github.com/org/repo"],
            "session_id": "s1",
        },
    )

    assert response.status_code == 400
    assert "session_id" in response.json()["detail"]
    assert not fake_remember


def test_repositories_without_code_content_type_rejected(client, fake_remember):
    response = client.post(
        "/remember",
        data={"datasetName": "ds", "repositories": ["https://github.com/org/repo"]},
    )

    assert response.status_code == 400
    assert "content_type='code'" in response.json()["detail"]
    assert not fake_remember


def test_index_vectors_without_code_content_type_rejected(client, fake_remember):
    response = client.post(
        "/remember",
        data={"datasetName": "ds", "index_vectors": "true"},
    )

    assert response.status_code == 400
    assert "index_vectors" in response.json()["detail"]
    assert not fake_remember


def test_local_paths_gated_by_accept_local_file_path(client, fake_remember, monkeypatch):
    monkeypatch.setattr(storage_module.settings, "accept_local_file_path", False)

    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "repositories": ["/srv/some/repo"],
        },
    )

    assert response.status_code == 400
    assert "ACCEPT_LOCAL_FILE_PATH" in response.json()["detail"]
    assert not fake_remember


def test_remote_urls_allowed_when_local_paths_disabled(client, fake_remember, monkeypatch):
    monkeypatch.setattr(storage_module.settings, "accept_local_file_path", False)

    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "repositories": ["https://github.com/org/repo"],
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == ["https://github.com/org/repo"]


def test_local_paths_allowed_by_default(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "repositories": ["/srv/some/repo"],
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == ["/srv/some/repo"]


def test_unknown_content_type_still_rejected(client, fake_remember):
    response = client.post(
        "/remember",
        data={"datasetName": "ds", "content_type": "sculpture"},
    )

    assert response.status_code == 400
    assert "Unsupported content_type" in response.json()["detail"]
    assert not fake_remember
