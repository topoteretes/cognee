"""Test: examples/guides/custom_tasks_and_pipelines.py

Verifies that the custom_tasks_and_pipelines guide runs end-to-end.
The guide defines a custom pipeline with an LLM extraction task
(``extract_people``) that calls ``LLMGateway.acreate_structured_output``
for each data item, then stores the results and runs ``cognee.cognify``.

``main()`` requires a text argument, so we pass a short example string.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_custom_tasks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_custom_tasks_and_pipelines_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """custom_tasks_and_pipelines.py should run its pipeline without LLM errors."""
    module = _load_example("examples/guides/custom_tasks_and_pipelines.py")

    assert hasattr(module, "main"), "custom_tasks_and_pipelines.py must expose a main() coroutine"
    try:
        await module.main("Alice knows Mark. Mark had dinner with Bob and Alice.")
    except (OSError, FileNotFoundError):
        # visualize_graph writes .artifacts/ dir which may not exist in CI
        pass
