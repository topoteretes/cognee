"""Test: examples/guides/graph_visualization.py

Verifies that the graph_visualization guide runs end-to-end.
The guide calls ``cognee.remember`` and then ``visualize_graph``, which
writes an HTML file.  OSError / FileNotFoundError from visualization are
tolerated (the ``.artifacts/`` directory may not exist in CI).
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_graph_viz", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_graph_visualization_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """graph_visualization.py should run remember() + visualize_graph() without LLM errors."""
    module = _load_example("examples/guides/graph_visualization.py")

    assert hasattr(module, "main"), "graph_visualization.py must expose a main() coroutine"
    try:
        await module.main()
    except (OSError, FileNotFoundError):
        pass
