"""The lexical (BM25) index contract driven by index_data_points and add_data_points (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tasks.storage.index_data_points import LEXICAL_BATCH_SIZE, index_data_points

adp_module = sys.modules["cognee.tasks.storage.add_data_points"]


class EmbeddedAndLexical(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"], "lexical_fields": ["text"]}


class LexicalOnly(DataPoint):
    text: str
    metadata: dict = {
        "index_fields": [],
        "lexical_fields": ["text"],
        "vector_only_fields": ["text"],
    }


def _engine(supports_lexical: bool = True, batch_size: int = 100):
    engine = SimpleNamespace(
        supports_lexical_search=supports_lexical,
        embedding_engine=SimpleNamespace(get_batch_size=lambda: batch_size),
        create_vector_index=AsyncMock(),
        index_data_points=AsyncMock(),
        create_lexical_index=AsyncMock(),
        index_lexical_data_points=AsyncMock(),
    )
    return engine


@pytest.mark.asyncio
async def test_an_embedded_lexical_field_is_indexed_after_its_rows_are_written():
    engine = _engine()
    points = [EmbeddedAndLexical(text="one"), EmbeddedAndLexical(text="two")]

    await index_data_points(points, vector_engine=engine)

    engine.create_vector_index.assert_awaited_once_with("EmbeddedAndLexical", "text")
    engine.index_data_points.assert_awaited_once()
    engine.create_lexical_index.assert_awaited_once_with("EmbeddedAndLexical", "text")
    engine.index_lexical_data_points.assert_awaited_once()
    args, kwargs = engine.index_lexical_data_points.await_args
    assert args[:2] == ("EmbeddedAndLexical", "text")
    assert [point.id for point in args[2]] == [point.id for point in points]
    assert kwargs["owns_rows"] is False, "the embedding path wrote the rows"


@pytest.mark.asyncio
async def test_a_lexical_only_field_owns_its_rows_and_is_never_embedded():
    engine = _engine()
    points = [LexicalOnly(text="cached document text")]

    await index_data_points(points, vector_engine=engine)

    engine.create_vector_index.assert_not_awaited()
    engine.index_data_points.assert_not_awaited()
    engine.create_lexical_index.assert_awaited_once_with("LexicalOnly", "text")
    _, kwargs = engine.index_lexical_data_points.await_args
    assert kwargs["owns_rows"] is True


@pytest.mark.asyncio
async def test_lexical_points_go_in_bounded_batches():
    engine = _engine()
    points = [LexicalOnly(text=f"doc {i}") for i in range(LEXICAL_BATCH_SIZE + 1)]

    await index_data_points(points, vector_engine=engine)

    assert engine.index_lexical_data_points.await_count == 2
    sizes = [len(call.args[2]) for call in engine.index_lexical_data_points.await_args_list]
    assert sizes == [LEXICAL_BATCH_SIZE, 1]


@pytest.mark.asyncio
async def test_a_backend_without_the_index_skips_the_lexical_step():
    engine = _engine(supports_lexical=False)

    await index_data_points(
        [EmbeddedAndLexical(text="one"), LexicalOnly(text="two")], vector_engine=engine
    )

    engine.index_data_points.assert_awaited_once()
    engine.create_lexical_index.assert_not_awaited()
    engine.index_lexical_data_points.assert_not_awaited()


def _unified(supports_lexical):
    graph_engine = AsyncMock()
    graph_engine.is_empty = AsyncMock(return_value=False)
    graph_engine.get_graph_metadata = AsyncMock(return_value={})
    vector_engine = MagicMock()
    vector_engine.supports_lexical_search = supports_lexical
    unified = AsyncMock()
    unified.graph = graph_engine
    unified.vector = vector_engine
    unified.has_capability = MagicMock(return_value=False)
    return unified, graph_engine


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_lexical", [True, False])
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_vector_only_fields_leave_the_graph_node_only_when_the_index_exists(
    mock_get_graph,
    mock_dedup,
    mock_get_unified,
    mock_index_nodes,
    mock_index_edges,
    supports_lexical,
):
    point = LexicalOnly(text="the full document text")
    mock_get_graph.side_effect = [([point], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine = _unified(supports_lexical)
    mock_get_unified.return_value = unified

    await add_data_points([point], ctx=PipelineContext())

    written = graph_engine.add_nodes.await_args.args[0]
    assert written[0].id == point.id
    if supports_lexical:
        assert written[0].text is None, "served from the lexical index, not the node"
    else:
        assert written[0].text == "the full document text", "the in-memory fallback reads the node"
    # The vector write always gets the text.
    embedded = mock_index_nodes.await_args.args[0]
    assert embedded[0].text == "the full document text"
