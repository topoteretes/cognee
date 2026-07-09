"""Mocked tests for examples/custom_pipelines/.

Each test loads the on-disk example and awaits its entrypoint under
isolated_example_env (mocked LLM + embeddings, per-test tmp_path), asserting it
runs to completion with no API key and no network.

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example

pytestmark = pytest.mark.asyncio


async def test_agentic_reasoning_procurement_example(isolated_example_env):
    # Entrypoint is run_procurement_example().
    module = import_example("examples/custom_pipelines/agentic_reasoning_procurement_example.py")
    await module.run_procurement_example()


async def test_custom_cognify_pipeline_example(isolated_example_env):
    module = import_example("examples/custom_pipelines/custom_cognify_pipeline_example.py")
    await module.main()


async def test_dynamic_steps_resume_analysis_hr_example(isolated_example_env):
    # main() takes a dict of enabled steps; run the full path.
    steps = {
        "prune_data": True,
        "prune_system": True,
        "add_text": True,
        "cognify": True,
        "graph_metrics": True,
        "retriever": True,
    }
    module = import_example("examples/custom_pipelines/dynamic_steps_resume_analysis_hr_example.py")
    await module.main(steps)


async def test_memify_coding_agent_rule_extraction_example(isolated_example_env):
    module = import_example(
        "examples/custom_pipelines/memify_coding_agent_rule_extraction_example.py"
    )
    await module.main()


async def test_organizational_hierarchy_pipeline_example(isolated_example_env):
    module = import_example(
        "examples/custom_pipelines/organizational_hierarchy/"
        "organizational_hierarchy_pipeline_example.py"
    )
    await module.main()


async def test_organizational_hierarchy_pipeline_low_level_example(isolated_example_env):
    module = import_example(
        "examples/custom_pipelines/organizational_hierarchy/"
        "organizational_hierarchy_pipeline_low_level_example.py"
    )
    await module.main()


async def test_relational_database_to_knowledge_graph_migration_example(isolated_example_env):
    module = import_example(
        "examples/custom_pipelines/relational_database_to_knowledge_graph_migration_example.py"
    )
    await module.main()
