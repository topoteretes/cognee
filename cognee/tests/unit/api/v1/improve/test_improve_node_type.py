"""Unit tests for how improve() resolves the node_type filter it hands to memify().

improve() defaults the filter to NodeSet, but it used to coerce an explicit
``node_type=None`` back to NodeSet too, so callers had no way to enrich without a
type filter. Omission and explicit None must now mean different things.
"""

import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.engine.models.node_set import NodeSet


@pytest.fixture
def memify_spy(monkeypatch):
    """Capture the kwargs improve() forwards to memify(), skipping every stage."""
    memify_module = importlib.import_module("cognee.modules.memify")
    captured = {}

    async def fake_memify(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(memify_module, "memify", fake_memify)

    users_methods = importlib.import_module("cognee.modules.users.methods")

    async def fake_get_default_user():
        return types.SimpleNamespace(id=uuid4())

    monkeypatch.setattr(users_methods, "get_default_user", fake_get_default_user)

    return captured


@pytest.mark.asyncio
async def test_node_type_defaults_to_node_set_when_omitted(memify_spy):
    from cognee.api.v1.improve import improve

    await improve(dataset="docs")

    assert memify_spy["node_type"] is NodeSet


@pytest.mark.asyncio
async def test_explicit_none_node_type_is_preserved(memify_spy):
    from cognee.api.v1.improve import improve

    await improve(dataset="docs", node_type=None)

    assert "node_type" in memify_spy
    assert memify_spy["node_type"] is None


@pytest.mark.asyncio
async def test_explicit_node_type_is_preserved(memify_spy):
    from cognee.api.v1.improve import improve

    class CustomNode:
        pass

    await improve(dataset="docs", node_type=CustomNode)

    assert memify_spy["node_type"] is CustomNode
