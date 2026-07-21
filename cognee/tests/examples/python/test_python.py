"""Mocked tests for examples/python/.

Each test loads the on-disk example and awaits its main() under
isolated_example_env (mocked LLM + embeddings, per-test tmp_path), asserting it
runs to completion with no API key and no network.

memory_provenance_demo, references_example, and schema_inventory_demo moved to
examples/demos/ on the current tree and are covered by the demos batch.

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example

pytestmark = pytest.mark.asyncio


async def test_truth_subspace_reranking_demo(isolated_example_env):
    module = import_example("examples/python/truth_subspace_reranking_demo.py")
    await module.main()
