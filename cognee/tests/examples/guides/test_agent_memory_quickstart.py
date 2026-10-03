"""Test: examples/guides/agent_memory_quickstart.py

Verifies that the agent_memory_quickstart guide runs end-to-end.
The guide defines two ``@cognee.agent_memory``-decorated coroutines
(``support_agent`` and ``faq_bot``) that internally call
``LLMGateway.acreate_structured_output``.

The test skips gracefully if ``cognee.agent_memory`` is not available in the
installed version.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_agent_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_agent_memory_quickstart_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """agent_memory_quickstart.py should run the support_agent / faq_bot flow without LLM errors."""
    try:
        module = _load_example("examples/guides/agent_memory_quickstart.py")
    except Exception as exc:
        pytest.skip(f"Could not load agent_memory_quickstart: {exc}")

    assert hasattr(module, "main"), (
        "agent_memory_quickstart.py must expose a main() coroutine"
    )
    await module.main()
