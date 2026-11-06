"""Real ingestion routes reject disabled fetching with an actionable HTTP error."""

import importlib
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.client import _stamp_operation_origin, exception_handler
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.exceptions import CogneeApiError
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.methods.get_default_user import get_default_user
from cognee.tasks.web_scraper.ssrf_protection import is_http_requests_allowed


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    app = FastAPI()
    app.middleware("http")(_stamp_operation_origin)
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.include_router(get_add_router(), prefix="/api/v1/add")
    app.include_router(get_remember_router(), prefix="/api/v1/remember")

    async def authenticated_user():
        return await get_default_user()

    app.dependency_overrides[get_authenticated_user] = authenticated_user
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("endpoint", ["add", "remember"])
@pytest.mark.parametrize("url", ["https://example.com/document", "https://github.com/org/repo"])
@pytest.mark.parametrize("setting", [None, "false"])
def test_ingestion_reports_how_to_enable_http_before_any_network_work(
    client, monkeypatch, endpoint, url, setting
):
    if setting is None:
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)
    storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")
    repo = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
    ssrf = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")
    fetch = AsyncMock(side_effect=AssertionError("Fetching must not start"))
    run_git = AsyncMock(side_effect=AssertionError("Git must not start"))
    resolve = AsyncMock(side_effect=AssertionError("DNS must not start"))
    monkeypatch.setattr(storage, "fetch_page_content", fetch)
    monkeypatch.setattr(repo, "_run_git", run_git)
    monkeypatch.setattr(ssrf, "_resolve_host_ips", resolve)

    response = client.post(
        f"/api/v1/{endpoint}",
        data={"datasetName": "http_policy", "raw_data": url, "self_improvement": "false"},
    )

    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert "ALLOW_HTTP_REQUESTS=true" in detail
    assert "on the server" in detail
    fetch.assert_not_awaited()
    run_git.assert_not_awaited()
    resolve.assert_not_awaited()
    if setting is None:
        # Merely importing/serving the API must not change in-process SDK calls.
        assert is_http_requests_allowed()


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
def test_text_and_uploads_are_accepted_with_http_disabled_by_default(client, monkeypatch, upload):
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
