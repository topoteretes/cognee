"""Test: examples/guides/consolidate_entity_descriptions_example.py

Verifies that the consolidate_entity_descriptions guide runs end-to-end.
The guide runs ``cognee.remember``, then
``consolidate_entity_descriptions_pipeline()``, both of which make LLM calls,
and finally two ``visualize_graph`` calls that write HTML files.

OSError / FileNotFoundError from the visualization step are tolerated (the
``.artifacts/`` directory may not exist in CI).
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_consolidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_consolidate_entity_descriptions_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """consolidate_entity_descriptions_example.py should run without LLM errors."""
    module = _load_example(
        "examples/guides/consolidate_entity_descriptions_example.py"
    )

    assert hasattr(module, "main"), (
        "consolidate_entity_descriptions_example.py must expose a main() coroutine"
    )
    try:
        await module.main()
    except (OSError, FileNotFoundError):
        pass
