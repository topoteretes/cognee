"""POST /api/v1/search and /recall with only_context: one string, and the retired
``context_format`` field is ignored rather than rejected (COG-6127).
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

# The routers package __init__ re-exports the same-named factory, so import by dotted path.
search_router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")

PROMPT = (
    "=== SYSTEM PROMPT ===\nAnswer the question using the provided context.\n\n"
    "=== USER PROMPT ===\nThe question is: `why?` ... node1 -- rel -- node2"
)


def test_search_endpoint_returns_the_only_context_string_and_drops_context_format(monkeypatch):
    app = FastAPI()
    app.include_router(search_router_module.get_search_router(), prefix="/api/v1/search")
    app.dependency_overrides[search_router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    monkeypatch.setattr(search_router_module, "send_telemetry", lambda *a, **k: None)

    captured = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return [
            {
                "search_result": PROMPT,
                "dataset_id": str(uuid4()),
                "dataset_name": "ds",
                "dataset_tenant_id": None,
            }
        ]

    import cognee.api.v1.search as search_pkg

    monkeypatch.setattr(search_pkg, "search", fake_search)

    response = TestClient(app).post(
        "/api/v1/search",
        json={
            "searchType": "GRAPH_COMPLETION",
            "query": "why?",
            "onlyContext": True,
            "sessionId": "s1",
            # A client written against 1.5.4 may still send this; it must be ignored.
            "contextFormat": "prompt",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()[0]["search_result"] == PROMPT
    assert captured["only_context"] is True
    assert captured["session_id"] == "s1"
    assert "context_format" not in captured


def test_request_dtos_ignore_the_retired_context_format_field():
    search_dto = search_router_module.SearchPayloadDTO(query="q", contextFormat="prompt")
    recall_dto = recall_router_module.RecallPayloadDTO(query="q", context_format="prompt")

    for dto in (search_dto, recall_dto):
        assert dto.only_context is False
        assert not hasattr(dto, "context_format")
        assert "context_format" not in dto.model_dump()
