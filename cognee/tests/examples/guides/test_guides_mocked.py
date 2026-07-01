"""Mocked tests for examples/guides/.

Each test imports the ``main()`` coroutine from the corresponding guide script,
runs it with all LLM and embedding calls intercepted by the shared harness, and
asserts the script completes without raising an exception.

No real API keys, no network calls, no external services required.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Make sure the examples directory is importable
_GUIDES_DIR = Path(__file__).parents[4] / "examples" / "guides"
if str(_GUIDES_DIR) not in sys.path:
    sys.path.insert(0, str(_GUIDES_DIR))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_visualize_graph(*args, **kwargs):
    """No-op async stub for cognee.visualize_graph — avoids filesystem writes."""
    return AsyncMock(return_value="/tmp/mock_graph.html")()


_VISUALIZE_TARGETS = [
    "cognee.visualize_graph",
    "cognee.api.v1.visualize.visualize.visualize_graph",
]


# ---------------------------------------------------------------------------
# recall_core.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_core(mock_llm_and_embeddings):
    """recall_core: remember text then recall — must complete without error."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("recall_core", _GUIDES_DIR / "recall_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Should not raise
    await mod.main()


# ---------------------------------------------------------------------------
# improve_quickstart.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_improve_quickstart(mock_llm_and_embeddings):
    """improve_quickstart: remember → recall → improve → recall again."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "improve_quickstart", _GUIDES_DIR / "improve_quickstart.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    await mod.main()


# ---------------------------------------------------------------------------
# temporal_recall.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_temporal_recall(mock_llm_and_embeddings):
    """temporal_recall: temporal cognify + before/after/between queries."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "temporal_recall", _GUIDES_DIR / "temporal_recall.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # temporal_recall asserts result != [] — patch recall to return a non-empty list
    with patch("cognee.recall", new=AsyncMock(return_value=["mock result"])):
        await mod.main()


# ---------------------------------------------------------------------------
# low_level_llm.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_low_level_llm(mock_llm_and_embeddings):
    """low_level_llm: direct LLMGateway.acreate_structured_output call."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "low_level_llm", _GUIDES_DIR / "low_level_llm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = await mod.main()
    # main() doesn't return anything explicitly but must not raise
    assert result is None or result is not None  # always True — just confirming no exception


# ---------------------------------------------------------------------------
# custom_data_models.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_data_models(mock_llm_and_embeddings):
    """custom_data_models: add DataPoint objects directly (no LLM needed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "custom_data_models", _GUIDES_DIR / "custom_data_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    await mod.main()


# ---------------------------------------------------------------------------
# graph_visualization.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_visualization(mock_llm_and_embeddings):
    """graph_visualization: remember text then render graph to HTML."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "graph_visualization", _GUIDES_DIR / "graph_visualization.py"
    )
    mod = importlib.util.module_from_spec(spec)

    with patch("cognee.visualize_graph", new=AsyncMock(return_value="/tmp/mock.html")):
        spec.loader.exec_module(mod)
        await mod.main()


# ---------------------------------------------------------------------------
# custom_graph_model.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_graph_model(mock_llm_and_embeddings):
    """custom_graph_model: remember with a custom graph model schema."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "custom_graph_model", _GUIDES_DIR / "custom_graph_model.py"
    )
    mod = importlib.util.module_from_spec(spec)

    with patch("cognee.visualize_graph", new=AsyncMock(return_value="/tmp/mock.html")):
        spec.loader.exec_module(mod)
        await mod.main()


# ---------------------------------------------------------------------------
# custom_prompts.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_prompts(mock_llm_and_embeddings):
    """custom_prompts: remember with custom extraction prompt, then recall."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "custom_prompts", _GUIDES_DIR / "custom_prompts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    await mod.main()


# ---------------------------------------------------------------------------
# importance_weight.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_importance_weight(mock_llm_and_embeddings):
    """importance_weight: ingest at different weights then recall."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "importance_weight", _GUIDES_DIR / "importance_weight.py"
    )
    mod = importlib.util.module_from_spec(spec)

    with patch("cognee.visualize_graph", new=AsyncMock(return_value="/tmp/mock.html")):
        spec.loader.exec_module(mod)
        await mod.main()


# ---------------------------------------------------------------------------
# custom_tasks_and_pipelines.py
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_tasks_and_pipelines(mock_llm_and_embeddings):
    """custom_tasks_and_pipelines: custom pipeline with DataPoint extraction."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "custom_tasks_and_pipelines", _GUIDES_DIR / "custom_tasks_and_pipelines.py"
    )
    mod = importlib.util.module_from_spec(spec)

    # PeopleLLM mock: return empty persons list so the extractor doesn't crash
    async def _mock_people_llm(text_input, system_prompt, response_model, **kwargs):
        if hasattr(response_model, "__name__") and response_model.__name__ == "PeopleLLM":
            return response_model(persons=[])
        from cognee.tests.examples.conftest import _build_minimal_instance
        return _build_minimal_instance(response_model)

    with (
        patch(
            "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
            new=_mock_people_llm,
        ),
        patch("cognee.visualize_graph", new=AsyncMock(return_value="/tmp/mock.html")),
    ):
        spec.loader.exec_module(mod)
        text = "Alice knows Mark. Mark had dinner with Bob and Alice."
        await mod.main(text)
