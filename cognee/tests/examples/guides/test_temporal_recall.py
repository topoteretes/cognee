"""Test: examples/guides/temporal_recall.py

Verifies that the temporal_recall guide runs end-to-end.
The guide calls ``cognee.remember`` with ``temporal_cognify=True`` and then
multiple ``cognee.recall`` calls using ``SearchType.TEMPORAL``.

The guide already contains ``assert result != []`` assertions — those will run
inside our mocked environment and should pass as long as the mocked LLM and
embedding layer return non-empty responses (which they do).
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_temporal_recall", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_temporal_recall_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """temporal_recall.py should run temporal remember() and recall() without LLM errors."""
    module = _load_example("examples/guides/temporal_recall.py")

    assert hasattr(module, "main"), "temporal_recall.py must expose a main() coroutine"
    await module.main()
