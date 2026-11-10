"""GET /api/v1/settings is superuser-only (SDK-1237).

The settings are server-wide (LLM and vector-DB providers, endpoints and key
prefixes), not the caller's own, so they are read by whoever may write them:
the POST already required a superuser, the GET did not.
"""

import importlib
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

router_module = importlib.import_module("cognee.api.v1.settings.routers.get_settings_router")


def _client(is_superuser: bool) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_settings_router(), prefix="/api/v1/settings")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id="u", is_superuser=is_superuser
    )
    return TestClient(app)


def test_get_settings_refuses_a_non_superuser(monkeypatch):
    called = False

    def fake_get_settings():
        nonlocal called
        called = True
        raise AssertionError("settings must not be read for a non-superuser")

    monkeypatch.setattr("cognee.modules.settings.get_settings", fake_get_settings)

    response = _client(is_superuser=False).get("/api/v1/settings")

    assert response.status_code == 403
    assert not called


def test_get_settings_serves_a_superuser(monkeypatch):
    fake = {
        "llm": {
            "provider": "openai",
            "model": "openai/gpt-5-mini",
            "endpoint": None,
            "api_version": None,
            "api_key": "sk-t********",
            "providers": [],
            "models": {},
        },
        "vector_db": {"provider": "lancedb", "url": "", "api_key": "", "providers": []},
    }
    monkeypatch.setattr("cognee.modules.settings.get_settings", lambda: fake)

    response = _client(is_superuser=True).get("/api/v1/settings")

    assert response.status_code == 200
    assert response.json()["llm"]["apiKey"] == "sk-t********"
