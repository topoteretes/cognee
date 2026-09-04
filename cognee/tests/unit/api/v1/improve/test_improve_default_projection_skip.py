"""Unit tests for the triplet-enrichment stage's projection-skip sentinel.

The default memify tasks never read the projected graph (they stream triplets
from the graph DB or no-op), so improve() passes the non-empty ``data=[{}]``
sentinel — the same one every other improve stage uses — to skip the
full-graph projection. Caller-supplied tasks/data must keep the projection,
since a custom task may consume the fragment.
"""

import importlib
import types
from unittest.mock import AsyncMock

import pytest

from cognee.modules.improve.stages import REASON_TRIPLET_EMBEDDING_DISABLED


@pytest.fixture
def memify_calls(monkeypatch, harness):
    """Stub the enrichment stage's dependencies and capture the memify call."""
    cognify_config_mod = importlib.import_module("cognee.modules.cognify.config")
    monkeypatch.setattr(
        cognify_config_mod,
        "get_cognify_config",
        lambda: types.SimpleNamespace(triplet_embedding=True),
    )
    changes_mod = importlib.import_module("cognee.modules.improve.graph_changes")
    monkeypatch.setattr(
        changes_mod, "has_graph_changed_since_last_improve", AsyncMock(return_value=True)
    )

    calls = []
    memify_mod = importlib.import_module("cognee.modules.memify")

    async def fake_memify(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(memify_mod, "memify", fake_memify)
    return calls


@pytest.mark.asyncio
async def test_default_call_skips_projection_via_sentinel(memify_calls, harness):
    result = await harness.improve()

    assert len(memify_calls) == 1
    assert memify_calls[0]["data"] == [{}]
    assert memify_calls[0]["dataset"] == harness.dataset.id  # resolved id, never the name
    assert memify_calls[0]["run_in_background"] is False
    assert result.stage("triplet_enrichment").status == "completed"


@pytest.mark.asyncio
async def test_custom_extraction_tasks_keep_projection(memify_calls, harness):
    custom_tasks = [object()]

    await harness.improve(extraction_tasks=custom_tasks)

    assert len(memify_calls) == 1
    assert memify_calls[0]["extraction_tasks"] == custom_tasks
    assert memify_calls[0].get("data") is None


@pytest.mark.asyncio
async def test_caller_supplied_data_is_preserved(memify_calls, harness):
    caller_data = ["my content"]

    await harness.improve(data=caller_data)

    assert len(memify_calls) == 1
    assert memify_calls[0]["data"] == caller_data


@pytest.mark.asyncio
async def test_triplet_embedding_off_skips_enrichment_without_calling_memify(monkeypatch, harness):
    cognify_config_mod = importlib.import_module("cognee.modules.cognify.config")
    monkeypatch.setattr(
        cognify_config_mod,
        "get_cognify_config",
        lambda: types.SimpleNamespace(triplet_embedding=False),
    )
    memify_mod = importlib.import_module("cognee.modules.memify")
    memify_mock = AsyncMock(return_value={})
    monkeypatch.setattr(memify_mod, "memify", memify_mock)

    result = await harness.improve()

    memify_mock.assert_not_awaited()
    stage = result.stage("triplet_enrichment")
    assert stage.status == "skipped"
    assert stage.reason == REASON_TRIPLET_EMBEDDING_DISABLED
    assert result.memify_run == {}


@pytest.mark.asyncio
async def test_unchanged_graph_reports_already_completed(monkeypatch, harness):
    cognify_config_mod = importlib.import_module("cognee.modules.cognify.config")
    monkeypatch.setattr(
        cognify_config_mod,
        "get_cognify_config",
        lambda: types.SimpleNamespace(triplet_embedding=True),
    )
    changes_mod = importlib.import_module("cognee.modules.improve.graph_changes")
    monkeypatch.setattr(
        changes_mod, "has_graph_changed_since_last_improve", AsyncMock(return_value=False)
    )
    memify_mod = importlib.import_module("cognee.modules.memify")
    memify_mock = AsyncMock(return_value={})
    monkeypatch.setattr(memify_mod, "memify", memify_mock)

    result = await harness.improve()

    memify_mock.assert_not_awaited()
    stage = result.stage("triplet_enrichment")
    assert stage.status == "already_completed"
    assert stage.reason == "no_writes_since_last_improve"
