"""The real add/remember routes refuse URL fetching with an actionable 403.

Runs the production ingestion path (route -> ``add()`` -> ``ingest_data`` ->
``save_data_item_to_storage`` / ``resolve_repo_source``) against isolated local
stores, with the network calls (page fetch, git, DNS) mocked so the test proves
the refusal fires before any of them. No cognify, no LLM.
"""

import asyncio
import importlib
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def policy_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_http_policy_test_"))

    import cognee  # cognee's import loads .env; the env below overrides it

    def clear_config_caches():
        for module_name, factory_name in [
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            (
                "cognee.infrastructure.databases.relational.get_relational_engine",
                "get_relational_engine",
            ),
            ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
            ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
            ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ]:
            try:
                getattr(importlib.import_module(module_name), factory_name).cache_clear()
            except (ImportError, AttributeError):
                pass

    mp = pytest.MonkeyPatch()
    for key, value in {
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "CACHE_BACKEND": "sqlite",
        "MOCK_EMBEDDING": "true",
        "TELEMETRY_DISABLED": "1",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    }.items():
        mp.setenv(key, value)
    clear_config_caches()

    # The routes resolve the default user before add() would create the database.
    from cognee.infrastructure.databases.relational import create_db_and_tables
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    asyncio.run(create_db_and_tables())
    # That engine is bound to the loop asyncio.run() just closed; the request
    # loops below must build their own.
    create_relational_engine.cache_clear()

    yield root

    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def client(policy_env):
    from cognee.api.client import exception_handler
    from cognee.api.v1.add.routers.get_add_router import get_add_router
    from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
    from cognee.exceptions import CogneeApiError
    from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
    from cognee.modules.users.methods.get_default_user import get_default_user

    app = FastAPI()
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.include_router(get_add_router(), prefix="/api/v1/add")
    app.include_router(get_remember_router(), prefix="/api/v1/remember")

    async def authenticated_user():
        return await get_default_user()

    app.dependency_overrides[get_authenticated_user] = authenticated_user
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def network_never_reached(monkeypatch):
    storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")
    repo = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
    ssrf = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")
    mocks = {
        "fetch": AsyncMock(side_effect=AssertionError("Fetching must not start")),
        "git": AsyncMock(side_effect=AssertionError("Git must not start")),
        "dns": AsyncMock(side_effect=AssertionError("DNS must not start")),
    }
    monkeypatch.setattr(storage, "fetch_page_content", mocks["fetch"])
    monkeypatch.setattr(repo, "_run_git", mocks["git"])
    monkeypatch.setattr(ssrf, "_resolve_host_ips", mocks["dns"])
    return mocks


@pytest.mark.parametrize("endpoint", ["add", "remember"])
@pytest.mark.parametrize("url", ["https://example.com/document", "https://github.com/org/repo"])
@pytest.mark.parametrize("setting", [None, "", "false"])
def test_ingestion_reports_how_to_enable_http_before_any_network_work(
    client, monkeypatch, network_never_reached, endpoint, url, setting
):
    if setting is None:
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)

    response = client.post(
        f"/api/v1/{endpoint}",
        data={"datasetName": "http_policy", "raw_data": url, "self_improvement": "false"},
    )

    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert "ALLOW_HTTP_REQUESTS=true" in detail
    assert "on the server" in detail
    # Forwarded as the policy's own message, not re-wrapped with an exception class name.
    assert "SSRFProtectionError:" not in detail
    for mock in network_never_reached.values():
        mock.assert_not_awaited()


def test_explicit_opt_in_allows_api_url_ingestion(client, monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")
    url = "https://8.8.8.8/document"
    fetch = AsyncMock(return_value={url: "<html><p>Public documentation.</p></html>"})
    monkeypatch.setattr(storage, "fetch_page_content", fetch)

    response = client.post("/api/v1/add", data={"datasetName": "http_opt_in", "raw_data": url})

    assert response.status_code == 200, response.text
    fetch.assert_awaited()


@pytest.mark.parametrize("upload", [False, True])
def test_text_and_uploads_are_accepted_with_http_disabled(client, monkeypatch, upload):
    monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")
    fetch = AsyncMock(side_effect=AssertionError("Text and uploads must not fetch URLs"))
    monkeypatch.setattr(storage, "fetch_page_content", fetch)
    form = {"datasetName": "http_text_and_uploads"}
    if upload:
        response = client.post(
            "/api/v1/add",
            data=form,
            files={"data": ("note.txt", b"An uploaded note.", "text/plain")},
        )
    else:
        response = client.post("/api/v1/add", data={**form, "raw_data": "A plain text note."})

    assert response.status_code == 200, response.text
    fetch.assert_not_awaited()
