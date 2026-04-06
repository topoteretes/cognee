import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.shared.data_models import KnowledgeGraph

cascade_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data_v2")


@pytest.mark.asyncio
@patch.object(cascade_module, "integrate_chunk_graphs", new_callable=AsyncMock)
@patch.object(cascade_module, "extract_edge_triplets", new_callable=AsyncMock)
@patch.object(
    cascade_module,
    "extract_content_nodes_and_relationship_names",
    new_callable=AsyncMock,
)
@patch.object(cascade_module, "extract_nodes", new_callable=AsyncMock)
async def test_extract_graph_from_data_passes_ontology_resolver(
    mock_extract_nodes,
    mock_extract_content_nodes_and_relationship_names,
    mock_extract_edge_triplets,
    mock_integrate_chunk_graphs,
):
    chunk = MagicMock(text="chunk text")
    ontology_resolver = MagicMock()

    mock_extract_nodes.return_value = ["node"]
    mock_extract_content_nodes_and_relationship_names.return_value = (
        ["node"],
        ["relationship"],
    )
    mock_extract_edge_triplets.return_value = MagicMock(name="chunk_graph")
    mock_integrate_chunk_graphs.return_value = [chunk]

    result = await cascade_module.extract_graph_from_data(
        [chunk], ontology_resolver=ontology_resolver
    )

    assert result == [chunk]
    mock_integrate_chunk_graphs.assert_awaited_once_with(
        data_chunks=[chunk],
        chunk_graphs=[mock_extract_edge_triplets.return_value],
        graph_model=KnowledgeGraph,
        ontology_resolver=ontology_resolver,
    )
