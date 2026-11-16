"""The chunker at the cognify() and remember() boundary and on POST /api/v1/cognify."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from cognee.api.client import app
from cognee.modules.chunking import chunker_registry
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.CsvChunker import CsvChunker
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.users.methods import get_authenticated_user

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


class ClauseChunker(Chunker):
    chunker_id = "clause_chunker_test"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(chunker_registry, "registered_chunkers", {})
    monkeypatch.delenv("DEFAULT_CHUNKER", raising=False)
    get_cognify_config.cache_clear()
    yield
    get_cognify_config.cache_clear()


def use_default(monkeypatch, name, chunker):
    chunker_registry.register_chunker(name, chunker)
    monkeypatch.setenv("DEFAULT_CHUNKER", name)
    get_cognify_config.cache_clear()


async def chunker_cognify_runs_with(**arguments):
    """The chunker cognify() hands to its task list."""
    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(cognify_module, "get_pipeline_executor") as executor,
        patch.object(
            cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[]
        ) as tasks,
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock),
        patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None),
    ):
        executor.return_value = AsyncMock(return_value={})
        # Named, so the task list does not depend on whether this machine has an LLM key.
        await cognify_module.cognify(extractor="llm", **arguments)
    return tasks.await_args.kwargs["chunker"]


@pytest.mark.asyncio
async def test_cognify_uses_text_chunker_when_nothing_names_another():
    assert await chunker_cognify_runs_with() is TextChunker


@pytest.mark.asyncio
async def test_cognify_uses_the_configured_default(monkeypatch):
    use_default(monkeypatch, "ClauseChunker", ClauseChunker)

    assert await chunker_cognify_runs_with() is ClauseChunker


@pytest.mark.asyncio
async def test_a_chunker_argument_wins_over_the_configured_default(monkeypatch):
    use_default(monkeypatch, "ClauseChunker", ClauseChunker)

    assert await chunker_cognify_runs_with(chunker=CsvChunker) is CsvChunker


@pytest.mark.asyncio
async def test_a_cognify_dry_run_estimates_with_the_configured_default(monkeypatch):
    use_default(monkeypatch, "ClauseChunker", ClauseChunker)

    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock),
        patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None),
        patch(
            "cognee.modules.cognify.estimator.estimate_cognify_dry_run", new_callable=AsyncMock
        ) as estimate,
    ):
        await cognify_module.cognify(dry_run=True, extractor="llm")

    assert estimate.await_args.kwargs["chunker"] is ClauseChunker


@pytest.mark.asyncio
async def test_a_remember_dry_run_estimates_with_the_configured_default(monkeypatch):
    use_default(monkeypatch, "ClauseChunker", ClauseChunker)

    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch(
            "cognee.modules.cognify.estimator.estimate_remember_dry_run", new_callable=AsyncMock
        ) as estimate,
    ):
        await remember_module.remember("some text", dry_run=True, extractor="llm")

    assert estimate.await_args.kwargs["chunker"] is ClauseChunker


def post_cognify(body):
    """POST /api/v1/cognify on the real app, so errors come back as clients see them."""
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=str(uuid4()))
    try:
        with (
            patch.object(
                importlib.import_module("cognee.api.v1.cognify"), "cognify", new_callable=AsyncMock
            ) as mock_cognify,
            TestClient(app) as client,
        ):
            mock_cognify.return_value = {}
            response = client.post("/api/v1/cognify", json={"datasets": ["contracts"], **body})
    finally:
        app.dependency_overrides.pop(get_authenticated_user, None)
    return response, mock_cognify


def test_the_endpoint_passes_a_named_chunker_as_its_class():
    chunker_registry.register_chunker("ClauseChunker", ClauseChunker)

    response, mock_cognify = post_cognify({"chunker": "ClauseChunker"})

    assert response.status_code == 200, response.text
    assert mock_cognify.await_args.kwargs["chunker"] is ClauseChunker


def test_the_endpoint_leaves_the_choice_to_the_server_when_no_chunker_is_named():
    response, mock_cognify = post_cognify({})

    assert response.status_code == 200, response.text
    assert mock_cognify.await_args.kwargs["chunker"] is None


@pytest.mark.parametrize("name", ["NoSuchChunker", "cognee.modules.chunking.TextChunker", ""])
def test_the_endpoint_refuses_a_name_that_is_not_registered(name):
    response, mock_cognify = post_cognify({"chunker": name})

    assert response.status_code == 400, response.text
    assert "Unknown chunker" in response.text and "TextChunker" in response.text
    mock_cognify.assert_not_awaited()
