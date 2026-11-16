"""The chunker cognify() and remember() use when a call passes none."""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking import chunker_registry
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.CsvChunker import CsvChunker
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


class ClauseChunker(Chunker):
    chunker_id = "clause_chunker_test"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(chunker_registry, "registered_chunkers", {"TextChunker": TextChunker})
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
