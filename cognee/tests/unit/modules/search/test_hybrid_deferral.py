from unittest.mock import AsyncMock, patch

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.search.methods.hybrid_deferral import (
    hybrid_deferral_reason,
    reject_hybrid_graph_only_knobs,
    request_deferral_reason,
)


class Entity:
    pass


def test_node_type_that_is_not_nodeset_defers():
    assert (
        request_deferral_reason({"node_name": ["KEN"], "node_type": Entity})
        == "node_type=Entity is not NodeSet"
    )


def test_custom_node_type_without_node_name_defers():
    assert request_deferral_reason({"node_type": Entity}) == "node_type=Entity is not NodeSet"


def test_neighborhood_depth_defers():
    assert request_deferral_reason({"neighborhood_depth": 2}) == "neighborhood_depth is set"


def test_positive_feedback_influence_defers():
    assert request_deferral_reason({"feedback_influence": 0.3}) == "feedback_influence > 0"


def test_graph_only_knobs_do_not_defer():
    assert request_deferral_reason({"wide_search_top_k": 200}) is None
    assert request_deferral_reason({"triplet_distance_penalty": 2.5}) is None


def test_explicit_wide_search_top_k_errors():
    with pytest.raises(CogneeValidationError, match="wide_search_top_k"):
        reject_hybrid_graph_only_knobs({"wide_search_top_k": 100})


def test_explicit_triplet_distance_penalty_errors():
    with pytest.raises(CogneeValidationError, match="triplet_distance_penalty"):
        reject_hybrid_graph_only_knobs({"triplet_distance_penalty": 6.5})


def test_omitted_graph_only_knobs_do_not_error():
    reject_hybrid_graph_only_knobs({})
    reject_hybrid_graph_only_knobs({"wide_search_top_k": None})
    reject_hybrid_graph_only_knobs({"triplet_distance_penalty": None})


def test_nodeset_scope_does_not_defer():
    """Hybrid filters 1-hop neighbours to the NodeSet; only a non-NodeSet type defers."""
    assert request_deferral_reason({"node_name": ["KEN"], "node_type": NodeSet}) is None


def test_other_clean_kwargs_do_not_defer():
    assert request_deferral_reason({}) is None
    assert request_deferral_reason({"wide_search_top_k": None}) is None
    assert request_deferral_reason({"node_type": None}) is None


def test_node_type_none_with_node_name_defers():
    assert (
        request_deferral_reason({"node_name": ["KEN"], "node_type": None})
        == "node_type=None is not NodeSet"
    )


@pytest.mark.asyncio
async def test_missing_document_chunk_collection_defers():
    engine = AsyncMock()
    engine.has_collection = AsyncMock(side_effect=lambda name: name != "DocumentChunk_text")

    with patch(
        "cognee.modules.search.methods.hybrid_deferral.get_vector_engine_async",
        new_callable=AsyncMock,
        return_value=engine,
    ):
        reason = await hybrid_deferral_reason({}, graph_is_empty=False)

    assert reason == "DocumentChunk_text collection missing"
    engine.has_collection.assert_awaited_once_with("DocumentChunk_text")


@pytest.mark.asyncio
async def test_missing_entity_collection_does_not_defer():
    engine = AsyncMock()
    engine.has_collection = AsyncMock(side_effect=lambda name: name != "Entity_name")

    with patch(
        "cognee.modules.search.methods.hybrid_deferral.get_vector_engine_async",
        new_callable=AsyncMock,
        return_value=engine,
    ):
        reason = await hybrid_deferral_reason({}, graph_is_empty=False)

    assert reason is None
    engine.has_collection.assert_awaited_once_with("DocumentChunk_text")


@pytest.mark.asyncio
async def test_empty_graph_skips_collection_checks():
    with patch(
        "cognee.modules.search.methods.hybrid_deferral.get_vector_engine_async",
        new_callable=AsyncMock,
    ) as get_engine:
        reason = await hybrid_deferral_reason({}, graph_is_empty=True)

    assert reason is None
    get_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_has_collection_error_fails_open():
    engine = AsyncMock()
    engine.has_collection = AsyncMock(side_effect=RuntimeError("backend missing method"))

    with patch(
        "cognee.modules.search.methods.hybrid_deferral.get_vector_engine_async",
        new_callable=AsyncMock,
        return_value=engine,
    ):
        reason = await hybrid_deferral_reason({}, graph_is_empty=False)

    assert reason is None
