"""Test: examples/guides/custom_graph_model.py

Verifies that the custom_graph_model guide runs without errors.
The guide calls ``cognee.remember`` with a custom ``graph_model`` and a
``custom_prompt``, which exercises the LLM extraction path.

Note: ``visualize_graph`` writes an HTML file to disk.  We skip the
visualization step to avoid filesystem side-effects in CI by only calling
``main()`` (which internally calls ``visualize_data()`` as well, so we
accept any ``FileNotFoundError`` or similar from the visualization step
and only fail on LLM/embedding errors).
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_custom_graph_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_custom_graph_model_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """custom_graph_model.py should run remember() + visualize without LLM errors."""
    module = _load_example("examples/guides/custom_graph_model.py")

    assert hasattr(module, "main"), "custom_graph_model.py must expose a main() coroutine"
    try:
        await module.main()
    except (OSError, FileNotFoundError):
        # Visualization writes to .artifacts/ dir which may not exist in CI — tolerated.
        pass
