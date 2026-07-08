"""Shared fixtures for mocked ``examples/`` tests.

The goal: run each example script end to end with zero secrets, no network,
and deterministic output. Three composable fixtures do this:

* ``mock_llm``        -- patches ``LLMGateway`` (structured output, transcript,
                         image) so no LLM provider is contacted.
* ``mock_embeddings`` -- flips cognee's built-in ``MOCK_EMBEDDING`` flag and
                         clears the cached embedding/vector engines. The real
                         engine class is still used (keeping its tokenizer);
                         only the network call is skipped.
* ``isolated_example_env`` -- points cognee's data/system roots at a per-test
                         ``tmp_path`` and prunes before/after, so tests can't
                         see each other's graph/vector/relational state or fight
                         over Ladybug/LanceDB file locks.

``isolated_example_env`` depends on the other two, so requesting it alone is
enough to get full isolation + mocking; the canaries list all three explicitly
for readability.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from cognee.tests.utils.example_mock_llm import patch_llm_gateway


def _clear_engine_caches() -> None:
    """Drop cached DB/embedding engines so new config/env takes effect."""
    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        create_embedding_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    for cached in (
        create_embedding_engine,
        _create_vector_engine,
        _create_graph_engine,
        create_relational_engine,
    ):
        try:
            cached.cache_clear()
        except Exception:
            pass


async def _prune_all() -> None:
    import cognee

    try:
        await cognee.prune.prune_data()
    except Exception:
        pass
    try:
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.fixture
def mock_llm():
    """Intercept all ``LLMGateway`` calls with deterministic canned responses."""
    with patch_llm_gateway():
        yield


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Return deterministic (zero) embeddings via cognee's MOCK_EMBEDDING flag."""
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    _clear_engine_caches()
    yield
    _clear_engine_caches()


@pytest_asyncio.fixture
async def isolated_example_env(tmp_path, monkeypatch, mock_llm, mock_embeddings):
    """Per-test data/system isolation with LLM + embedding mocks applied."""
    import cognee

    # Provider/config env that keeps everything local and key-free.
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("LLM_API_KEY", "mock-key-for-testing")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    data_dir = tmp_path / "data"
    system_dir = tmp_path / "system"
    data_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    # Point cognee at the tmp roots. system_root_directory cascades to the
    # relational/graph/vector path configs.
    cognee.config.data_root_directory(str(data_dir))
    cognee.config.system_root_directory(str(system_dir))

    _clear_engine_caches()
    await _prune_all()

    yield tmp_path

    await _prune_all()
    _clear_engine_caches()
