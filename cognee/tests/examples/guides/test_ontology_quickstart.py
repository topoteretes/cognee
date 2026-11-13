"""Test: examples/guides/ontology_quickstart.py

Verifies that the ontology_quickstart guide runs end-to-end.
The guide loads a local OWL ontology file and calls ``cognee.remember``
with an ``RDFLibOntologyResolver``.

Note: The test requires the ontology file at
``examples/guides/ontology_input_example/basic_ontology.owl`` to be
present in the workspace (it is part of the repo).
"""

import importlib.util
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_ontology_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_ontology_quickstart_runs_without_error(
    mock_cognee_llm, mock_cognee_embeddings, clean_cognee_state
):
    """ontology_quickstart.py should run remember() with ontology config without LLM errors."""
    ontology_file = (
        Path(__file__).parents[4]
        / "examples/guides/ontology_input_example/basic_ontology.owl"
    )
    if not ontology_file.exists():
        pytest.skip("Ontology file not found — skipping ontology quickstart test.")

    module = _load_example("examples/guides/ontology_quickstart.py")

    assert hasattr(module, "main"), "ontology_quickstart.py must expose a main() coroutine"
    await module.main()
