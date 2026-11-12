"""POST /api/v1/search and /recall forward retriever_specific_config to the SDK call,
the way the SDK's own search()/recall() already accept it. The known keys are checked
for type and upper bound, and the one key JSON cannot carry (response_model) is rejected.
A rejected request answers 400: the app maps every request validation error to 400."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from cognee.api.client import request_validation_exception_handler

search_router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")

CONFIG = {
    "chunks_top_k": 20,
    "include_external_metadata": True,
    "external_metadata_keys": ["source_id", "title"],
}
# A value that must never appear in telemetry.
SENTINEL = "metadata-key-sentinel-7f3a"


def _app(router_module, name):
    """An app serving ``get_<name>_router()`` under ``/api/v1/<name>`` for a stub user.

    It registers the production validation handler, so a rejected body answers with the
    status a real client gets.
    """
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.include_router(getattr(router_module, f"get_{name}_router")(), prefix=f"/api/v1/{name}")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return app


@pytest.fixture
def search_client(monkeypatch):
    captured = {}
    app = _app(search_router_module, "search")
    telemetry = {}

    def fake_send_telemetry(event_name, user=None, additional_properties=None, **kwargs):
        telemetry.update(additional_properties or {})

    monkeypatch.setattr(search_router_module, "send_telemetry", fake_send_telemetry)

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    import cognee.api.v1.search as search_pkg

    monkeypatch.setattr(search_pkg, "search", fake_search)
    return TestClient(app), captured, telemetry


@pytest.fixture
def recall_client(monkeypatch):
    captured = {}
    app = _app(recall_router_module, "recall")
    monkeypatch.setattr(recall_router_module, "send_telemetry", lambda *a, **k: None)

    async def fake_recall(**kwargs):
        captured.update(kwargs)
        return []

    # ``cognee.api.v1.recall`` the attribute is the function; the router imports from
    # the package module, so patch that.
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.recall"), "recall", fake_recall)
    return TestClient(app), captured


def test_search_forwards_retriever_specific_config(search_client):
    client, captured, telemetry = search_client

    response = client.post(
        "/api/v1/search",
        json={
            "query": "what did Acme order?",
            "retrieverSpecificConfig": {**CONFIG, "external_metadata_keys": [SENTINEL]},
        },
    )

    assert response.status_code == 200, response.text
    assert captured["retriever_specific_config"] == {
        **CONFIG,
        "external_metadata_keys": [SENTINEL],
    }
    # Telemetry records the shape (number of keys), never the values.
    assert telemetry["retriever_specific_config"] == len(CONFIG)
    assert SENTINEL not in str(telemetry)


def test_search_without_the_field_sends_none(search_client):
    client, captured, _ = search_client

    response = client.post("/api/v1/search", json={"query": "q"})

    assert response.status_code == 200, response.text
    assert captured["retriever_specific_config"] is None


def test_search_rejects_response_model_in_config(search_client):
    client, captured, _ = search_client

    response = client.post(
        "/api/v1/search",
        json={"query": "q", "retrieverSpecificConfig": {"response_model": "str"}},
    )

    assert response.status_code == 400, response.text
    assert "response_model" in response.text
    assert captured == {}


def test_recall_forwards_retriever_specific_config(recall_client):
    client, captured = recall_client

    response = client.post(
        "/api/v1/recall",
        json={"query": "what did Acme order?", "retriever_specific_config": CONFIG},
    )

    assert response.status_code == 200, response.text
    assert captured["retriever_specific_config"] == CONFIG
    # response_schema stays the structured-output channel; nothing else is folded in.
    assert captured["response_model"] is None


def test_recall_rejects_response_model_in_config(recall_client):
    client, captured = recall_client

    response = client.post(
        "/api/v1/recall",
        json={"query": "q", "retrieverSpecificConfig": {"response_model": "str"}},
    )

    assert response.status_code == 400, response.text
    assert "response_schema" in response.text
    assert captured == {}


INVALID_CONFIGS = [
    # Resource knobs above their ceiling.
    ({"store_mb": 1e9}, "store_mb"),
    ({"batch_mb": 1e9}, "batch_mb"),
    ({"context_tokens": 10**12}, "context_tokens"),
    ({"reading_tokens": 10**9}, "reading_tokens"),
    ({"chunks_top_k": 100_000}, "chunks_top_k"),
    ({"entities_top_k": 100_000}, "entities_top_k"),
    ({"facts_top_k": 100_000}, "facts_top_k"),
    ({"max_iter": 500}, "max_iter"),
    # Values the retrievers reject or misread.
    ({"store_mb": 0}, "store_mb"),
    ({"max_iter": 0}, "max_iter"),
    ({"max_iter": None}, "max_iter"),
    ({"chunks_top_k": -1}, "chunks_top_k"),
    ({"chunks_top_k": "20"}, "chunks_top_k"),
    ({"chunks_top_k": 2.5}, "chunks_top_k"),
    ({"chunks_top_k": True}, "chunks_top_k"),
    ({"include_external_metadata": "yes"}, "include_external_metadata"),
    # A bare string would otherwise be split into characters.
    ({"external_metadata_keys": "title"}, "external_metadata_keys"),
    ({"external_metadata_keys": ["title", 3]}, "external_metadata_keys"),
    ({"system_prompt_path": 3}, "system_prompt_path"),
]


@pytest.mark.parametrize("config, key", INVALID_CONFIGS)
def test_search_rejects_invalid_config_values(search_client, config, key):
    client, captured, _ = search_client

    response = client.post("/api/v1/search", json={"query": "q", "retrieverSpecificConfig": config})

    assert response.status_code == 400, response.text
    assert key in response.text
    assert captured == {}


@pytest.mark.parametrize("config, key", INVALID_CONFIGS)
def test_recall_rejects_invalid_config_values(recall_client, config, key):
    client, captured = recall_client

    response = client.post("/api/v1/recall", json={"query": "q", "retrieverSpecificConfig": config})

    assert response.status_code == 400, response.text
    assert key in response.text
    assert captured == {}


def test_values_at_the_bounds_and_unknown_keys_pass(search_client):
    client, captured, _ = search_client
    config = {
        "chunks_top_k": 1_000,
        "entities_top_k": 0,
        "facts_top_k": None,
        "max_iter": 20,
        "store_mb": 256,
        "batch_mb": 0.5,
        "context_tokens": 1_000_000,
        "external_metadata_keys": None,
        "a_key_no_retriever_reads": {"anything": [1, "two"]},
    }

    response = client.post("/api/v1/search", json={"query": "q", "retrieverSpecificConfig": config})

    assert response.status_code == 200, response.text
    assert captured["retriever_specific_config"] == config
