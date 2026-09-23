"""Wiring test for POST /api/v1/search telemetry event properties (COG-6244).

Pins the property set of the "Search API Endpoint Invoked" event to
shape-only request descriptors, matching the recall endpoint's convention
(query_length instead of query, etc.). A regression that reintroduces
request payload values into the event fails here.
"""

import importlib
import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

# The routers package __init__ re-exports the same-named factory function, so
# the module must be imported by its dotted path.
router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")

QUERY_SENTINEL = "SENTINEL-query-do-not-export"
PROMPT_SENTINEL = "SENTINEL-prompt-do-not-export"
NODE_SENTINEL = "SENTINEL-node-do-not-export"
CODE_SENTINEL = "SENTINEL-code-do-not-export"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router_module.get_search_router(), prefix="/api/v1/search")
    return app


def test_search_event_carries_shape_only_properties(monkeypatch):
    app = _app()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )

    captured = {}

    def fake_send_telemetry(event_name, user=None, additional_properties=None, **kwargs):
        captured["event"] = event_name
        captured["properties"] = additional_properties or {}

    monkeypatch.setattr(router_module, "send_telemetry", fake_send_telemetry)

    async def fake_search(**kwargs):
        return []

    import cognee.api.v1.search as search_pkg

    monkeypatch.setattr(search_pkg, "search", fake_search)

    client = TestClient(app)
    response = client.post(
        "/api/v1/search",
        json={
            "searchType": "CHUNKS",
            "query": QUERY_SENTINEL,
            "systemPrompt": PROMPT_SENTINEL,
            "nodeName": [NODE_SENTINEL],
            "topK": 7,
        },
    )
    assert response.status_code == 200, response.text

    properties = captured["properties"]

    # The original property keys stay, carrying sizes instead of values.
    assert properties["query"] == len(QUERY_SENTINEL)
    assert properties["system_prompt"] == len(PROMPT_SENTINEL)
    assert properties["node_name"] == 1
    assert properties["code_query"] == 0
    assert properties["top_k"] == 7

    # No request payload values anywhere in the event.
    serialized = json.dumps(properties, default=str)
    for sentinel in (QUERY_SENTINEL, PROMPT_SENTINEL, NODE_SENTINEL, CODE_SENTINEL):
        assert sentinel not in serialized
