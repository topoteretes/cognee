"""Test: examples/guides/improve_quickstart.py

Verifies that the improve_quickstart guide runs end-to-end.
The guide calls ``cognee.remember`` (twice), ``cognee.recall``,
``cognee.improve`` (which triggers LLM-based graph enrichment), and a
second ``cognee.recall``.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_improve_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_improve_quickstart_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """improve_quickstart.py should run remember() / improve() / recall() without LLM errors."""
    module = _load_example("examples/guides/improve_quickstart.py")

    assert hasattr(module, "main"), "improve_quickstart.py must expose a main() coroutine"
    await module.main()
