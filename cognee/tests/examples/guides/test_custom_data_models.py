"""Test: examples/guides/custom_data_models.py

Verifies that the custom_data_models guide runs end-to-end.
The guide creates DataPoint instances (Person) and stores them via
``add_data_points`` — no LLM call, but embeddings are used for vector
indexing.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_custom_data_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_custom_data_models_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """custom_data_models.py should add data points without errors."""
    module = _load_example("examples/guides/custom_data_models.py")

    assert hasattr(module, "main"), "custom_data_models.py must expose a main() coroutine"
    await module.main()
