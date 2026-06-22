import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from starlette.datastructures import Headers

router_module = importlib.import_module("cognee.api.v1.cognify.routers.get_cognify_router")


class WebSocketStub:
    def __init__(self, *, headers=None, cookies=None):
        self.headers = Headers(headers or {})
        self.cookies = cookies or {}


def test_websocket_access_token_prefers_bearer_header():
    websocket = WebSocketStub(
        headers={"Authorization": "Bearer header-token"},
        cookies={"auth_token": "cookie-token"},
    )

    assert router_module._get_websocket_access_token(websocket) == "header-token"


def test_websocket_access_token_falls_back_to_cookie():
    websocket = WebSocketStub(
        headers={"Authorization": "Basic ignored"},
        cookies={"auth_token": "cookie-token"},
    )

    assert router_module._get_websocket_access_token(websocket) == "cookie-token"


@pytest.mark.asyncio
async def test_websocket_auth_fetches_active_user_from_jwt(monkeypatch):
    token_user = SimpleNamespace(id=uuid4(), is_active=True)
    hydrated_user = SimpleNamespace(id=token_user.id, is_active=True)
    read_user = AsyncMock(return_value=token_user)
    get_user = AsyncMock(return_value=hydrated_user)
    get_default_user = AsyncMock()

    monkeypatch.setattr(router_module, "REQUIRE_AUTHENTICATION", True)
    monkeypatch.setattr(router_module, "_read_websocket_jwt_user", read_user)
    monkeypatch.setattr(router_module, "get_user", get_user)
    monkeypatch.setattr(router_module, "get_default_user", get_default_user)

    websocket = WebSocketStub(headers={"Authorization": "Bearer jwt-token"})
    user = await router_module._get_websocket_authenticated_user(websocket)

    assert user is hydrated_user
    read_user.assert_awaited_once_with("jwt-token")
    get_user.assert_awaited_once_with(token_user.id)
    get_default_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_websocket_auth_rejects_inactive_user(monkeypatch):
    token_user = SimpleNamespace(id=uuid4(), is_active=False)
    read_user = AsyncMock(return_value=token_user)
    get_user = AsyncMock()

    monkeypatch.setattr(router_module, "REQUIRE_AUTHENTICATION", True)
    monkeypatch.setattr(router_module, "_read_websocket_jwt_user", read_user)
    monkeypatch.setattr(router_module, "get_user", get_user)

    websocket = WebSocketStub(headers={"Authorization": "Bearer jwt-token"})
    user = await router_module._get_websocket_authenticated_user(websocket)

    assert user is None
    get_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_websocket_auth_uses_default_user_if_auth_disabled(monkeypatch):
    default_user = SimpleNamespace(id=uuid4(), is_active=True)
    read_user = AsyncMock(return_value=None)
    get_default_user = AsyncMock(return_value=default_user)

    monkeypatch.setattr(router_module, "REQUIRE_AUTHENTICATION", False)
    monkeypatch.setattr(router_module, "_read_websocket_jwt_user", read_user)
    monkeypatch.setattr(router_module, "get_default_user", get_default_user)

    user = await router_module._get_websocket_authenticated_user(WebSocketStub())

    assert user is default_user
    read_user.assert_awaited_once_with(None)
    get_default_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_authorized_cognify_pipeline_run_requires_dataset_read(monkeypatch):
    dataset_id = uuid4()
    pipeline_run_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    pipeline_run = SimpleNamespace(pipeline_run_id=pipeline_run_id, dataset_id=dataset_id)
    get_pipeline_run = AsyncMock(return_value=pipeline_run)
    get_authorized_dataset = AsyncMock(return_value=SimpleNamespace(id=dataset_id))

    monkeypatch.setattr(router_module, "get_pipeline_run", get_pipeline_run)
    monkeypatch.setattr(router_module, "get_authorized_dataset", get_authorized_dataset)

    result = await router_module._get_authorized_cognify_pipeline_run(str(pipeline_run_id), user)

    assert result is pipeline_run
    get_pipeline_run.assert_awaited_once_with(pipeline_run_id)
    get_authorized_dataset.assert_awaited_once_with(user, dataset_id, "read")


@pytest.mark.asyncio
async def test_authorized_cognify_pipeline_run_rejects_missing_read_access(monkeypatch):
    dataset_id = uuid4()
    pipeline_run_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    pipeline_run = SimpleNamespace(pipeline_run_id=pipeline_run_id, dataset_id=dataset_id)

    monkeypatch.setattr(router_module, "get_pipeline_run", AsyncMock(return_value=pipeline_run))
    monkeypatch.setattr(router_module, "get_authorized_dataset", AsyncMock(return_value=None))

    result = await router_module._get_authorized_cognify_pipeline_run(str(pipeline_run_id), user)

    assert result is None


@pytest.mark.asyncio
async def test_authorized_cognify_pipeline_run_rejects_invalid_run_id(monkeypatch):
    get_pipeline_run = AsyncMock()
    monkeypatch.setattr(router_module, "get_pipeline_run", get_pipeline_run)

    result = await router_module._get_authorized_cognify_pipeline_run(
        "not-a-uuid", SimpleNamespace(id=uuid4())
    )

    assert result is None
    get_pipeline_run.assert_not_awaited()
