"""Tool calls on POST /api/v1/responses run as the authenticated caller.

The router authenticates the request and hands that user to
``dispatch_function``; every dispatched operation (search, cognify/add) runs
with the caller's permissions. ``prune`` is not dispatchable at all: it is an
instance-wide system wipe with no per-user scope, and nothing a model-chosen
tool call may reach.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.responses.dispatch_function import dispatch_function
from cognee.api.v1.responses.routers.get_responses_router import get_responses_router
from cognee.modules.users.methods import get_authenticated_user

dispatch_module = importlib.import_module("cognee.api.v1.responses.dispatch_function")
router_module = importlib.import_module("cognee.api.v1.responses.routers.get_responses_router")


def _caller():
    return SimpleNamespace(id=uuid4(), email="caller@example.com", is_active=True, tenant_id=None)


def _tool_call(name, **arguments):
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.mark.asyncio
async def test_search_runs_as_the_given_user(monkeypatch):
    caller = _caller()
    search = AsyncMock(return_value=["hit"])
    monkeypatch.setattr(dispatch_module, "search", search)

    result = await dispatch_function(_tool_call("search", search_query="what is x"), caller)

    assert result == ["hit"]
    assert search.await_args.kwargs["user"] is caller


@pytest.mark.asyncio
async def test_cognify_runs_as_the_given_user(monkeypatch):
    caller = _caller()
    add = AsyncMock()
    cognify = AsyncMock()
    monkeypatch.setattr(dispatch_module, "add", add)
    monkeypatch.setattr(dispatch_module, "cognify", cognify)

    await dispatch_function(_tool_call("cognify", text="remember this"), caller)

    assert add.await_args.kwargs["user"] is caller
    assert cognify.await_args.kwargs["user"] is caller


@pytest.mark.asyncio
async def test_prune_is_not_dispatchable():
    result = await dispatch_function(_tool_call("prune"), _caller())

    assert result == "Error: Unknown function prune"
    assert not hasattr(dispatch_module, "prune")
    assert not hasattr(dispatch_module, "get_default_user")


def test_router_dispatches_as_the_authenticated_caller(monkeypatch):
    caller = _caller()
    dispatched = AsyncMock(return_value="ok")
    monkeypatch.setattr(router_module, "dispatch_function", dispatched)
    monkeypatch.setattr(router_module, "get_llm_config", lambda: SimpleNamespace(llm_api_key="k"))

    class FakeResponses:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                model_dump=lambda: {
                    "id": "resp_1",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "search",
                            "arguments": json.dumps({"search_query": "q"}),
                            "call_id": "call_1",
                        }
                    ],
                    "usage": {},
                }
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr(router_module.openai, "AsyncOpenAI", FakeOpenAI)

    app = FastAPI()
    app.include_router(get_responses_router(), prefix="/api/v1/responses")
    app.dependency_overrides[get_authenticated_user] = lambda: caller

    response = TestClient(app).post(
        "/api/v1/responses/", json={"input": "search for q", "model": "cognee-v1"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["toolCalls"][0]["output"]["status"] == "success"
    assert dispatched.await_count == 1
    assert dispatched.await_args.args[1] is caller
