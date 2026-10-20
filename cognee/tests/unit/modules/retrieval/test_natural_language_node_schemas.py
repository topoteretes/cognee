"""The live node-label schema must reach the Cypher-generation prompt.

Regression for the case where ``_execute_cypher_query`` computed ``node_schemas``
but dropped them before calling ``_generate_cypher_query`` (only ``edge_schemas``
was threaded through), while the prompt template hardcoded a static node-schema
block. The effect was that the LLM never saw the store's real node labels, so
every generated Cypher query targeted guessed labels and bound nothing.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever


@pytest.mark.asyncio
async def test_live_node_labels_reach_generation_prompt():
    # A label that is NOT part of the template's old static block, so it can only
    # appear in the rendered prompt if the live node schema is actually threaded in.
    node_schema = [{"NodeLabels": ["WidgetGizmo"], "Properties": ["sku", "name"]}]
    edge_schema = [{"key": "USED_IN"}]
    raw_result = [{"n": {"id": "w-1", "name": "Sprocket"}}]

    mock_engine = AsyncMock()
    mock_engine.query = AsyncMock(side_effect=[node_schema, edge_schema, raw_result])

    captured = {}

    async def _capture(text_input, system_prompt, response_model):
        captured["system_prompt"] = system_prompt
        return "MATCH (n:WidgetGizmo) RETURN n"

    retriever = NaturalLanguageRetriever(max_attempts=1)
    with patch(
        "cognee.modules.retrieval.natural_language_retriever.LLMGateway.acreate_structured_output",
        AsyncMock(side_effect=_capture),
    ):
        await retriever._execute_cypher_query("find widgets", mock_engine)

    # The real template + render_prompt ran; the live label must be present.
    assert "WidgetGizmo" in captured["system_prompt"], (
        "live node labels were not threaded into the Cypher-generation prompt"
    )
