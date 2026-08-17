import pytest
from unittest.mock import AsyncMock, patch

from cognee.modules.retrieval.connection_path_retriever import (
    ConnectionPathRetriever,
    NO_PATH_MESSAGE,
    _STRUCTURAL_NODE_TYPES,
)


def _triplet(src, rel, dst):
    return ({"id": src, "name": src}, {"relationship_name": rel}, {"id": dst, "name": dst})


@pytest.mark.asyncio
async def test_context_renders_ordered_chain():
    retriever = ConnectionPathRetriever()
    edges = retriever._path_to_edges(
        [_triplet("raj", "works_at", "acme"), _triplet("acme", "operates", "apollo")]
    )

    context = await retriever.get_context_from_objects(retrieved_objects=edges)

    assert "raj --[works_at]--> acme" in context
    assert "acme --[operates]--> apollo" in context


@pytest.mark.asyncio
async def test_context_no_path_message_when_empty():
    retriever = ConnectionPathRetriever(max_depth=4)
    context = await retriever.get_context_from_objects(retrieved_objects=[])
    assert context == NO_PATH_MESSAGE.format(max_depth=4)


@pytest.mark.asyncio
async def test_completion_returns_no_path_message_without_llm_call():
    retriever = ConnectionPathRetriever()
    no_path = NO_PATH_MESSAGE.format(max_depth=5)

    with patch(
        "cognee.modules.retrieval.connection_path_retriever.generate_completion",
        new=AsyncMock(),
    ) as mocked_completion:
        result = await retriever.get_completion_from_context(
            query="how is A connected to B?",
            retrieved_objects=[],
            context=no_path,
        )

    assert result == [no_path]
    mocked_completion.assert_not_called()


@pytest.mark.asyncio
async def test_get_retrieved_objects_uses_anchors_and_find_paths():
    retriever = ConnectionPathRetriever()

    graph_engine = AsyncMock()
    graph_engine.is_empty = AsyncMock(return_value=False)
    graph_engine.find_paths = AsyncMock(
        return_value=[[_triplet("raj", "works_at", "acme"), _triplet("acme", "operates", "apollo")]]
    )

    vector_engine = AsyncMock()
    # Two different resolved node ids for the two anchors.
    vector_engine.search = AsyncMock(
        side_effect=[
            [type("R", (), {"id": "raj"})()],
            [type("R", (), {"id": "apollo"})()],
        ]
    )

    with (
        patch(
            "cognee.modules.retrieval.connection_path_retriever.get_graph_engine",
            new=AsyncMock(return_value=graph_engine),
        ),
        patch(
            "cognee.modules.retrieval.connection_path_retriever.get_vector_engine",
            return_value=vector_engine,
        ),
        patch.object(
            ConnectionPathRetriever,
            "_extract_anchors",
            new=AsyncMock(return_value=("raj", "apollo")),
        ),
    ):
        edges = await retriever.get_retrieved_objects(query="how is raj connected to apollo?")

    graph_engine.find_paths.assert_awaited_once_with(
        "raj", "apollo", max_depth=5, excluded_node_types=_STRUCTURAL_NODE_TYPES
    )
    assert [e.node1.attributes["name"] for e in edges] == ["raj", "acme"]
    assert edges[-1].node2.attributes["name"] == "apollo"
