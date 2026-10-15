"""CODE search rejections reach REST clients with a stable ``code`` and ``param``.

The app-level CogneeApiError handler copies a CodeSearchValidationError's
``error_code`` and ``error_param`` into the 422 body next to ``detail``, so a
client can tell an older server (``unknown_operation``) from a seed this
dataset does not hold (``not_found``) without parsing the message. Through
/v1/recall a seed miss is "no code facts" instead (an empty result), while a
bad request still answers 422.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.recall.routers.get_recall_router import get_recall_router
from cognee.api.v1.search.routers.get_search_router import get_search_router
from cognee.exceptions import CogneeApiError
from cognee.modules.retrieval.code_retriever import (
    CodeRetriever,
    CodeSearchValidationError,
    invalidate_code_graph_snapshot_cache,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_authenticated_user

search_pkg = importlib.import_module("cognee.api.v1.search")
search_methods = importlib.import_module("cognee.modules.search.methods.search")

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=None)
GRAPH = (
    [
        (
            "handler",
            {
                "name": "api/orders.create_order",
                "type": "CodeSymbol",
                "file_path": "api/orders.py",
                "line": 3,
                "repo": "shop",
            },
        )
    ],
    [],
)


@pytest.fixture
def client(monkeypatch):
    from cognee.api.client import exception_handler

    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    invalidate_code_graph_snapshot_cache(all_entries=True)

    async def search_code(*, query_text, code_query, **_kwargs):
        # The real retriever over a stub graph, so the router and the handler
        # see exactly the errors a CODE search raises.
        return [await CodeRetriever(config=code_query).get_retrieved_objects(query_text)]

    async def recall_code(*, query_text, retriever_specific_config, **_kwargs):
        # recall's code lane calls authorized_search with SearchType.CODE pinned.
        result = await CodeRetriever(config=retriever_specific_config).get_retrieved_objects(
            query_text
        )
        return [
            SearchResultPayload(completion=result, search_type=SearchType.CODE, dataset_id=uuid4())
        ]

    monkeypatch.setattr(search_pkg, "search", search_code)
    monkeypatch.setattr(search_methods, "authorized_search", recall_code)
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=GRAPH)

    app = FastAPI()
    app.include_router(get_search_router(), prefix="/api/v1/search")
    app.include_router(get_recall_router(), prefix="/api/v1/recall")
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.dependency_overrides[get_authenticated_user] = lambda: MOCK_USER
    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        yield TestClient(app)
    invalidate_code_graph_snapshot_cache(all_entries=True)


def _search(client, code_query):
    return client.post(
        "/api/v1/search",
        json={"query": "seed", "search_type": "CODE", "code_query": code_query},
    )


def _recall(client, code_query):
    return client.post(
        "/api/v1/recall",
        json={
            "query": "seed",
            "scope": ["code"],
            "dataset_ids": [str(uuid4())],
            "code_query": code_query,
        },
    )


def test_unknown_operation_answers_422_with_its_code(client):
    response = _search(client, {"operation": "ask_the_oracle"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "unknown_operation"
    assert body["param"] == "operation"
    assert body["detail"].startswith("Unsupported CODE operation 'ask_the_oracle'")
    assert body["detail"].endswith("[CodeSearchValidationError]")


def test_missing_seed_answers_422_with_not_found_and_the_param(client):
    response = _search(client, {"operation": "neighbors", "name": "payments.refund"})

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "CODE could not resolve neighbors anchor name 'payments.refund'. "
            "[CodeSearchValidationError]"
        ),
        "code": "not_found",
        "param": "name",
    }


def test_invalid_argument_and_success_through_the_router(client):
    rejected = _search(client, {"operation": "search", "text": "orders", "limit": -1})
    assert rejected.status_code == 422
    assert (rejected.json()["code"], rejected.json()["param"]) == ("invalid_argument", "limit")

    accepted = _search(client, {"operation": "search", "text": "create order"})
    assert accepted.status_code == 200
    (result,) = accepted.json()
    assert result["operation"] == "search"
    assert [hit["id"] for hit in result["hits"]] == ["handler"]


def test_recall_code_scope_rejects_bad_requests_and_drops_seed_misses(client):
    rejected = _recall(client, {"operation": "ask_the_oracle"})
    assert rejected.status_code == 422
    assert (rejected.json()["code"], rejected.json()["param"]) == (
        "unknown_operation",
        "operation",
    )

    missed = _recall(client, {"operation": "neighbors", "name": "payments.refund"})
    assert missed.status_code == 200
    assert missed.json() == []

    answered = _recall(client, {"operation": "capabilities"})
    assert answered.status_code == 200
    (entry,) = answered.json()
    assert entry["source"] == "code"
    assert entry["raw"]["operation"] == "capabilities"
    assert entry["raw"]["counts"]["nodes"] == 1


@pytest.mark.asyncio
async def test_handler_adds_only_string_codes():
    from cognee.api.client import exception_handler

    with pytest.raises(CodeSearchValidationError) as raised:
        CodeRetriever(config={"operation": "search", "diagram": True})
    coded = json.loads((await exception_handler(None, raised.value)).body)
    assert (coded["code"], coded["param"]) == ("invalid_argument", "diagram")

    without_param = CodeSearchValidationError("CODE cannot do that.")
    assert set(json.loads((await exception_handler(None, without_param)).body)) == {
        "detail",
        "code",
    }

    # Other errors may carry an unrelated, non-string error_code (the CLI's
    # exit status, for one); the body must not pick it up.
    other = CogneeApiError("plain", "Plain", status_code=400, log=False)
    other.error_code = 2
    other.error_param = ""
    assert set(json.loads((await exception_handler(None, other)).body)) == {"detail"}
