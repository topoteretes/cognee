"""Unit tests for improve() stage 3's projection-skip sentinel.

The default memify tasks never read the projected graph (they stream triplets
from the graph DB or no-op), so improve() passes the non-empty ``data=[{}]``
sentinel — the same one every other improve stage uses — to skip the
full-graph projection. Caller-supplied tasks/data must keep the projection,
since a custom task may consume the fragment.
"""

import importlib
import types
from uuid import uuid4

import pytest


def _make_user():
    return types.SimpleNamespace(id=uuid4())


@pytest.fixture
def improve_mod():
    return importlib.import_module("cognee.api.v1.improve.improve")


@pytest.fixture
def memify_calls(monkeypatch, improve_mod):
    """Stub out everything improve() needs before stage 3 and capture the memify call."""
    user = _make_user()
    dataset = types.SimpleNamespace(id=uuid4(), owner_id=user.id)

    async def fake_resolve(_dataset, _user):
        return _user, [dataset]

    monkeypatch.setattr(improve_mod, "resolve_authorized_user_datasets", fake_resolve)

    state_mod = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(state_mod, "get_remote_client", lambda: None)

    utils_mod = importlib.import_module("cognee.shared.utils")
    monkeypatch.setattr(utils_mod, "send_telemetry", lambda *args, **kwargs: None)

    calls = []
    memify_mod = importlib.import_module("cognee.modules.memify")

    async def fake_memify(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(memify_mod, "memify", fake_memify)
    return calls


@pytest.mark.asyncio
async def test_default_call_skips_projection_via_sentinel(memify_calls):
    improve_mod = importlib.import_module("cognee.api.v1.improve.improve")

    await improve_mod.improve(user=_make_user())

    assert len(memify_calls) == 1
    assert memify_calls[0]["data"] == [{}]


@pytest.mark.asyncio
async def test_custom_extraction_tasks_keep_projection(memify_calls):
    improve_mod = importlib.import_module("cognee.api.v1.improve.improve")
    custom_tasks = [object()]

    await improve_mod.improve(user=_make_user(), extraction_tasks=custom_tasks)

    assert len(memify_calls) == 1
    assert memify_calls[0]["extraction_tasks"] == custom_tasks
    assert memify_calls[0].get("data") is None


@pytest.mark.asyncio
async def test_caller_supplied_data_is_preserved(memify_calls):
    improve_mod = importlib.import_module("cognee.api.v1.improve.improve")
    caller_data = ["my content"]

    await improve_mod.improve(user=_make_user(), data=caller_data)

    assert len(memify_calls) == 1
    assert memify_calls[0]["data"] == caller_data
