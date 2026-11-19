"""Test: examples/guides/recall_core.py

Verifies that the recall_core guide runs end-to-end with mocked LLM and
embedding backends.  The guide calls ``cognee.remember`` (which triggers
graph-building + embedding) and ``cognee.recall`` (which queries the graph and
runs LLM completions).
"""

import importlib.util
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------


def _load_example(rel_path: str):
    """Dynamically load an example script as an isolated module."""
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_recall_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_core_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """recall_core.py should run its main() end-to-end with mocked backends."""
    module = _load_example("examples/guides/recall_core.py")

    assert hasattr(module, "main"), "recall_core.py must expose a main() coroutine"
    # main() does not return a meaningful value — just ensure no exception is raised
    await module.main()
