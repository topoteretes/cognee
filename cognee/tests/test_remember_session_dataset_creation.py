"""Regression test for issue #4089.

A session-mode ``remember()`` (``session_id`` set, ``self_improvement=True``)
launches a background ``improve()`` that bridges the session into the target
dataset. Before the fix the dataset was never created, so every background
bridge stage failed on write/read authorization. This test asserts the
dataset is created/authorized *before* the background improve runs — no LLM
required (``improve`` is stubbed).
"""

from pathlib import Path

import pytest
import pytest_asyncio

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user


async def _reset_and_prune() -> None:
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest_asyncio.fixture
async def clean_env(tmp_path):
    root = Path(tmp_path)
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))
    await _reset_and_prune()
    await engine_setup()
    try:
        yield
    finally:
        await _reset_and_prune()


@pytest.mark.asyncio
async def test_session_remember_creates_dataset_before_background_improve(clean_env, monkeypatch):
    dataset_name = "agent_scoped_memory"

    # Stub improve so the test needs no LLM, and record whether the target
    # dataset already exists (is authorized) at the moment improve is invoked.
    # remember() does `from cognee.api.v1.improve import improve`, so patch the
    # name on that package module (the dotted path resolves to the re-exported
    # function, so fetch the package object explicitly).
    import importlib

    seen = {}

    async def fake_improve(dataset, *, session_ids=None, user=None, **kwargs):
        resolver = user or await get_default_user()
        existing = await get_authorized_existing_datasets([dataset], "write", resolver)
        seen["dataset_present_at_improve"] = bool(existing)
        return {}

    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    monkeypatch.setattr(improve_pkg, "improve", fake_improve)

    result = await cognee.remember(
        "The launch codename is Saffron.",
        dataset_name=dataset_name,
        session_id="clean-first-session",
        self_improvement=True,
    )

    # remember() creates/authorizes the dataset synchronously, before it
    # returns and before the background improve runs.
    user = await get_default_user()
    assert await get_datasets_by_name([dataset_name], user.id), (
        "session remember must create the target dataset up front"
    )

    # Drain the background improve and confirm it saw an existing dataset.
    await result
    assert seen.get("dataset_present_at_improve") is True
