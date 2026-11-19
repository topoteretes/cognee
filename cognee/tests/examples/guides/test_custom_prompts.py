"""Test: examples/guides/custom_prompts.py

Verifies that the custom_prompts guide runs end-to-end.
The guide calls ``cognee.remember`` with a custom_prompt and then
``cognee.recall`` with GRAPH_COMPLETION search type, exercising the LLM
extraction + retrieval path.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_custom_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_custom_prompts_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """custom_prompts.py should run remember() + recall() without LLM errors."""
    module = _load_example("examples/guides/custom_prompts.py")

    assert hasattr(module, "main"), "custom_prompts.py must expose a main() coroutine"
    await module.main()
