"""Test: examples/guides/low_level_llm.py

Verifies that the low_level_llm guide runs end-to-end with a mocked LLM.
The guide directly calls ``LLMGateway.acreate_structured_output`` and expects a
``MiniGraph`` instance back.
"""

import importlib.util
from pathlib import Path

import pytest
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_low_level_llm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_level_llm_runs_without_error(mock_cognee_llm, mock_cognee_embeddings):
    """low_level_llm.py should call main() and get a structured response back."""
    module = _load_example("examples/guides/low_level_llm.py")

    assert hasattr(module, "main"), "low_level_llm.py must expose a main() coroutine"
    # main() calls LLMGateway.acreate_structured_output(text, system_prompt, MiniGraph)
    # Our mock returns make_fake_structured_response(MiniGraph) which is a MiniGraph instance.
    await module.main()
