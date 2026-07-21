"""Mocked tests for examples/guides/.

Each test loads the on-disk example and awaits its main() under
isolated_example_env (mocked LLM + embeddings, per-test tmp_path), asserting it
runs to completion with no API key and no network.

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example, requires_aws

pytestmark = pytest.mark.asyncio


async def test_agent_memory_quickstart(isolated_example_env):
    module = import_example("examples/guides/agent_memory_quickstart.py")
    await module.main()


@pytest.mark.xfail(
    reason=(
        "cognee's consolidate_entity_descriptions memify pipeline raises "
        "KeyError('id') in get_entities_with_neighborhood on the minimal mocked "
        "graph. Needs a richer mock graph or upstream handling; tracked as a "
        "harness follow-up on #3958."
    ),
    strict=False,
)
async def test_consolidate_entity_descriptions_example(isolated_example_env):
    module = import_example("examples/guides/consolidate_entity_descriptions_example.py")
    await module.main()


async def test_custom_data_models(isolated_example_env):
    module = import_example("examples/guides/custom_data_models.py")
    await module.main()


@pytest.mark.xfail(
    reason=(
        "The mocked KnowledgeGraph leaves the example's optional "
        "'used_in: List[Field] = None' field as None, and cognee re-validates the "
        "extracted graph against the full model which requires a list. Needs "
        "build_mock_response (harness, #3958) to populate list-typed fields with "
        "[]; raising as a harness improvement with the author."
    ),
    strict=False,
)
async def test_custom_graph_model(isolated_example_env):
    module = import_example("examples/guides/custom_graph_model.py")
    await module.main()


async def test_custom_prompts(isolated_example_env):
    module = import_example("examples/guides/custom_prompts.py")
    await module.main()


async def test_custom_tasks_and_pipelines(isolated_example_env):
    # This example's main() takes the input text as an argument (its __main__
    # passes a sample sentence); mirror that so the pipeline runs end to end.
    module = import_example("examples/guides/custom_tasks_and_pipelines.py")
    await module.main("Alice knows Mark. Mark had dinner with Bob and Alice. Bob knows Mary.")


async def test_graph_visualization(isolated_example_env):
    module = import_example("examples/guides/graph_visualization.py")
    await module.main()


async def test_importance_weight(isolated_example_env):
    module = import_example("examples/guides/importance_weight.py")
    await module.main()


async def test_improve_quickstart(isolated_example_env):
    module = import_example("examples/guides/improve_quickstart.py")
    await module.main()


async def test_low_level_llm(isolated_example_env):
    module = import_example("examples/guides/low_level_llm.py")
    await module.main()


async def test_ontology_quickstart(isolated_example_env):
    module = import_example("examples/guides/ontology_quickstart.py")
    await module.main()


async def test_recall_core(isolated_example_env):
    module = import_example("examples/guides/recall_core.py")
    await module.main()


@requires_aws()
async def test_s3_storage(isolated_example_env):
    # Talks to a real s3:// bucket; skipped unless AWS credentials are present.
    module = import_example("examples/guides/s3_storage.py")
    await module.main()


async def test_temporal_recall(isolated_example_env):
    module = import_example("examples/guides/temporal_recall.py")
    await module.main()
