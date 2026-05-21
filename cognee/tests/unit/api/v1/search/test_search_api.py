import types
from uuid import uuid4, uuid5, NAMESPACE_OID

import pytest

from cognee.modules.search.types import SearchType


def _make_user(user_id: str = "u1", tenant_id=None):
    return types.SimpleNamespace(id=user_id, tenant_id=tenant_id)


def _make_dataset(*, name="ds", tenant_id="t1", dataset_id=None, owner_id=None):
    return types.SimpleNamespace(
        id=dataset_id or uuid5(NAMESPACE_OID, name),
        name=name,
        tenant_id=uuid5(NAMESPACE_OID, tenant_id),
        owner_id=owner_id or uuid4(),
    )


@pytest.fixture
def api_search_mod():
    import importlib

    return importlib.import_module("cognee.api.v1.search.search")


@pytest.mark.asyncio
async def test_api_graph_search_passes_feedback_influence_to_search_function(
    monkeypatch, api_search_mod
):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["feedback_influence"] == 0.4
        return ["ok"]

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    out = await api_search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        dataset_ids=[dataset.id],
        feedback_influence=0.4,
    )

    assert out == ["ok"]


@pytest.mark.asyncio
async def test_api_graph_search_uses_updated_default_triplet_penalty(monkeypatch, api_search_mod):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["triplet_distance_penalty"] == 6.5
        return ["ok"]

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    out = await api_search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        dataset_ids=[dataset.id],
    )

    assert out == ["ok"]
