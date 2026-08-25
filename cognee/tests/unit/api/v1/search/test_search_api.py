import types
from uuid import uuid4, uuid5, NAMESPACE_OID

import pytest

from cognee.exceptions import CogneeValidationError
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
async def test_api_graph_search_omits_unspecified_triplet_penalty(monkeypatch, api_search_mod):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["triplet_distance_penalty"] is None
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


@pytest.mark.asyncio
async def test_api_code_search_merges_code_query_into_retriever_config(monkeypatch, api_search_mod):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["retriever_specific_config"] == {
            "existing": True,
            "operation": "find_path",
            "target": "PaymentStore",
        }
        return [{"found": True}]

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    result = await api_search_mod.search(
        query_text="CheckoutService",
        query_type=SearchType.CODE,
        user=user,
        dataset_ids=[dataset.id],
        retriever_specific_config={"existing": True},
        code_query={"operation": "find_path", "target": "PaymentStore"},
    )

    assert result == [{"found": True}]


@pytest.mark.asyncio
async def test_api_code_query_rejects_non_code_search(api_search_mod):
    with pytest.raises(CogneeValidationError, match="code_query requires"):
        await api_search_mod.search(
            query_text="CheckoutService",
            query_type=SearchType.CHUNKS,
            user=_make_user(),
            code_query={"operation": "explore"},
        )


@pytest.mark.asyncio
async def test_api_search_rejects_unknown_context_format(api_search_mod):
    with pytest.raises(CogneeValidationError, match="context_format"):
        await api_search_mod.search(
            query_text="why?",
            query_type=SearchType.GRAPH_COMPLETION,
            user=_make_user(),
            only_context=True,
            context_format="bogus",
        )


@pytest.mark.asyncio
async def test_api_search_forwards_context_format(monkeypatch, api_search_mod):
    captured = {}

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        captured["context_format"] = kwargs.get("context_format")
        return []

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    await api_search_mod.search(
        query_text="why?",
        query_type=SearchType.GRAPH_COMPLETION,
        user=_make_user(),
        dataset_ids=[_make_dataset().id],
        only_context=True,
        context_format="prompt",
    )

    assert captured["context_format"] == "prompt"


def test_search_dto_publishes_and_enforces_the_context_format_enum():
    from pydantic import ValidationError

    from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO
    from cognee.modules.search.types import ContextFormat

    assert (
        SearchPayloadDTO(query="q", contextFormat="prompt").context_format is ContextFormat.PROMPT
    )
    assert SearchPayloadDTO(query="q").context_format is ContextFormat.CONTEXT
    with pytest.raises(ValidationError):
        SearchPayloadDTO(query="q", contextFormat="bogus")


def test_search_dto_carries_session_id_in_both_casings():
    """/v1/search previously dropped the session_id the cloud client already sends."""
    from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO

    assert SearchPayloadDTO(query="q", sessionId="s1").session_id == "s1"
    assert SearchPayloadDTO(query="q", session_id="s1").session_id == "s1"
    assert SearchPayloadDTO(query="q").session_id is None
