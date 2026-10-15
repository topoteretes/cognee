"""The /v1/remember router's code repository options.

Repositories take the normal ingestion path: content_type="code" is accepted
and ignored, and codegraph_config (a JSON form field) reaches remember()
validated. ``remember()`` itself is monkeypatched on the
``cognee.api.v1.remember`` package (the router imports it lazily at call
time), so no databases, git, enola, or network are involved.
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


def test_code_content_type_is_ignored_and_specs_are_forwarded(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "content_type": "code",
            "raw_data": ["https://github.com/org/repo", "", "https://github.com/org/other"],
            "run_in_background": "true",
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["data"] == [
        "https://github.com/org/repo",
        "https://github.com/org/other",
    ]
    assert fake_remember["kwargs"]["content_type"] is None
    assert "codegraph_config" not in fake_remember["kwargs"]
    assert fake_remember["kwargs"]["run_in_background"] is True


def test_codegraph_config_is_parsed_and_forwarded(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "raw_data": "https://git.example.com/org/repo",
            "codegraph_config": (
                '{"treat_as_repository": true, "include_documents": false, "index_vectors": null}'
            ),
        },
    )

    assert response.status_code == 200, response.text
    # None keys are dropped by validation.
    assert fake_remember["kwargs"]["codegraph_config"] == {
        "treat_as_repository": True,
        "include_documents": False,
    }


def test_legacy_index_vectors_field_is_forwarded_for_folding(client, fake_remember):
    response = client.post(
        "/remember",
        data={
            "datasetName": "code_ds",
            "raw_data": "https://github.com/org/repo",
            "index_vectors": "true",
        },
    )

    assert response.status_code == 200, response.text
    assert fake_remember["kwargs"]["index_vectors"] is True


@pytest.mark.parametrize(
    "value",
    [
        "not json",
        '["a list"]',
        '{"unknown_key": true}',
        '{"include_documents": "false"}',
    ],
)
def test_invalid_codegraph_config_is_a_400(client, fake_remember, value):
    response = client.post(
        "/remember",
        data={"datasetName": "code_ds", "raw_data": "/repo", "codegraph_config": value},
    )

    assert response.status_code == 400, response.text
    assert "codegraph_config" in response.json()["detail"]
    assert not fake_remember
