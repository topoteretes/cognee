"""Test: examples/guides/importance_weight.py

Verifies that the importance_weight guide runs end-to-end.
The guide calls ``cognee.remember`` three times with different
``importance_weight`` values, then ``visualize_graph`` and ``cognee.recall``.

OSError / FileNotFoundError from the visualization step are tolerated.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_importance_weight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_importance_weight_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """importance_weight.py should run remember() / recall() without LLM errors."""
    module = _load_example("examples/guides/importance_weight.py")

    assert hasattr(module, "main"), "importance_weight.py must expose a main() coroutine"
    try:
        await module.main()
    except (OSError, FileNotFoundError):
        pass
