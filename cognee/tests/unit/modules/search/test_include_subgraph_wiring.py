import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.types import SearchType
from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload


def test_get_retriever_output_attaches_subgraph_when_requested():
    edge = MagicMock()
    edge.node1 = MagicMock(id="n1", attributes={"name": "A", "type": "Entity"})
    edge.node2 = MagicMock(id="n2", attributes={"name": "B", "type": "Entity"})
    edge.attributes = {"relationship_name": "rel", "vector_distance": [0.5]}

    retriever = MagicMock()
    retriever.prepare_session_turn_for_retrieval = AsyncMock(
        return_value=types.SimpleNamespace(should_answer=True, effective_query="q")
    )
    retriever.get_retrieved_objects = AsyncMock(return_value=[edge])
    retriever.get_context_from_objects = AsyncMock(return_value="ctx")
    retriever.get_completion_from_context = AsyncMock(return_value="answer")

    async def _run():
        with (
            patch(
                "cognee.modules.search.methods.get_retriever_output.get_search_type_retriever_instance",
                AsyncMock(return_value=retriever),
            ),
            patch(
                "cognee.modules.search.methods.get_retriever_output.get_graph_engine"
            ) as graph_engine,
            patch(
                "cognee.modules.search.methods.get_retriever_output.update_node_access_timestamps",
                AsyncMock(),
            ),
        ):
            graph_engine.return_value.is_empty = AsyncMock(return_value=False)
            return await get_retriever_output(
                SearchType.GRAPH_COMPLETION,
                "question",
                include_subgraph=True,
            )

    payload = asyncio.run(_run())
    assert payload.retrieved_subgraph is not None
    assert payload.retrieved_subgraph["edges"][0]["source"] == "n1"
    assert payload.retrieved_subgraph["edges"][0]["target"] == "n2"


def test_get_retriever_output_omits_subgraph_by_default():
    retriever = MagicMock()
    retriever.prepare_session_turn_for_retrieval = AsyncMock(
        return_value=types.SimpleNamespace(should_answer=True, effective_query="q")
    )
    retriever.get_retrieved_objects = AsyncMock(return_value=[])
    retriever.get_context_from_objects = AsyncMock(return_value="")
    retriever.get_completion_from_context = AsyncMock(return_value="answer")

    async def _run():
        with (
            patch(
                "cognee.modules.search.methods.get_retriever_output.get_search_type_retriever_instance",
                AsyncMock(return_value=retriever),
            ),
            patch(
                "cognee.modules.search.methods.get_retriever_output.get_graph_engine"
            ) as graph_engine,
            patch(
                "cognee.modules.search.methods.get_retriever_output.update_node_access_timestamps",
                AsyncMock(),
            ),
        ):
            graph_engine.return_value.is_empty = AsyncMock(return_value=False)
            return await get_retriever_output(
                SearchType.GRAPH_COMPLETION,
                "question",
            )

    payload = asyncio.run(_run())
    assert payload.retrieved_subgraph is None


def test_normalize_search_payload_non_graph_include_subgraph_explicit_null():
    payload = SearchResultPayload(
        completion="chunk text",
        search_type=SearchType.CHUNKS,
        retrieved_subgraph=None,
    )
    items = normalize_search_payload(payload, include_subgraph=True)
    assert len(items) == 1
    assert items[0].retrieved_subgraph is None


def test_normalize_search_payload_default_excludes_subgraph_field():
    payload = SearchResultPayload(
        completion="answer",
        search_type=SearchType.GRAPH_COMPLETION,
        retrieved_subgraph={"nodes": [], "edges": []},
    )
    items = normalize_search_payload(payload, include_subgraph=False)
    assert "retrieved_subgraph" not in items[0].model_fields_set


def test_backwards_compatible_search_results_default_unchanged(search_mod, monkeypatch):
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    payload = SearchResultPayload(
        completion="answer",
        search_type=SearchType.GRAPH_COMPLETION,
    )
    out = search_mod._backwards_compatible_search_results([payload], verbose=False)
    out_with_flag = search_mod._backwards_compatible_search_results(
        [payload], verbose=False, include_subgraph=False
    )
    assert out == out_with_flag
    assert "retrieved_subgraph" not in (out if isinstance(out, dict) else {})


def test_backwards_compatible_search_results_wraps_when_subgraph_requested(search_mod, monkeypatch):
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    payload = SearchResultPayload(
        completion="answer",
        search_type=SearchType.GRAPH_COMPLETION,
        retrieved_subgraph={"nodes": [], "edges": []},
    )
    out = search_mod._backwards_compatible_search_results(
        [payload], verbose=False, include_subgraph=True
    )
    assert out == {"search_result": "answer", "retrieved_subgraph": {"nodes": [], "edges": []}}


@pytest.fixture
def search_mod():
    import importlib

    return importlib.import_module("cognee.modules.search.methods.search")
