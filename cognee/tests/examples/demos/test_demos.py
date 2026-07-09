"""Mocked tests for examples/demos/.

Each test loads the on-disk example and awaits its entrypoint under
isolated_example_env (mocked LLM + embeddings, per-test tmp_path), asserting it
runs to completion with no API key and no network.

Most demos expose async def main(); exceptions (alt entrypoints, args, external
services) are handled and commented per-test.

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example, invoke_example_main

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Standard demos: async def main(), no args, no external service.
# ---------------------------------------------------------------------------


async def test_agentic_session_context_demo(isolated_example_env):
    # Uses argparse; invoke_example_main keeps sys.argv pytest-safe at call time.
    rel = "examples/demos/agentic_session_context_demo.py"
    module = import_example(rel)
    await invoke_example_main(module, rel)


async def test_comprehensive_example(isolated_example_env):
    module = import_example("examples/demos/comprehensive_example/cognee_comprehensive_example.py")
    await module.main()


async def test_conversation_session_persistence_example(isolated_example_env):
    module = import_example("examples/demos/conversation_session_persistence_example.py")
    await module.main()


async def test_custom_graph_model_entity_schema_definition(isolated_example_env):
    module = import_example("examples/demos/custom_graph_model_entity_schema_definition.py")
    await module.main()


async def test_custom_pipeline_single_object_example(isolated_example_env):
    module = import_example("examples/demos/custom_pipeline_single_object_example.py")
    await module.main()


@pytest.mark.skip(
    reason="Requires the optional dlt extra (pip install cognee[dlt]); not in the base keyless env."
)
async def test_dlt_ingestion_example(isolated_example_env):
    module = import_example("examples/demos/dlt_ingestion_example.py")
    await module.main()


async def test_dynamic_multiple_weighted_edges_example(isolated_example_env):
    module = import_example("examples/demos/dynamic_multiple_weighted_edges_example.py")
    await module.main()


async def test_feedback_score_shifting_example(cached_example_env):
    module = import_example("examples/demos/feedback_score_shifting_example.py")
    await module.main()


async def test_global_context_index_smoke_demo(isolated_example_env):
    module = import_example("examples/demos/global_context_index_smoke_demo.py")
    await module.main()


async def test_live_session_context_feedback_demo(isolated_example_env):
    # Uses argparse; invoke_example_main keeps sys.argv pytest-safe at call time.
    rel = "examples/demos/live_session_context_feedback_demo.py"
    module = import_example(rel)
    await invoke_example_main(module, rel)


@pytest.mark.xfail(
    reason=(
        "Hits an LLM path not covered by the harness (a raw completion object "
        "with .choices is expected, but the mock returns a str). Still fails on "
        "#3958's b5c57d704 hardening (its JSON/.choices mock did not reach this "
        "call site); tracked as a further harness follow-up."
    ),
    strict=False,
)
async def test_memory_provenance_demo(isolated_example_env):
    module = import_example("examples/demos/memory_provenance_demo.py")
    await module.main()


async def test_multimedia_audio_image_processing_example(isolated_example_env):
    module = import_example(
        "examples/demos/multimedia_processing/multimedia_audio_image_processing_example.py"
    )
    await module.main()


async def test_nodeset_grouping_example(isolated_example_env):
    module = import_example("examples/demos/nodeset_grouping_example.py")
    await module.main()


async def test_ontology_as_reference_vocabulary_example(isolated_example_env):
    module = import_example(
        "examples/demos/ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py"
    )
    await module.main()


async def test_references_example(isolated_example_env):
    module = import_example("examples/demos/references_example.py")
    await module.main()


async def test_remember_recall_improve_example(isolated_example_env):
    module = import_example("examples/demos/remember_recall_improve_example.py")
    await module.main()


async def test_schema_inventory_demo(isolated_example_env):
    module = import_example("examples/demos/schema_inventory_demo.py")
    await module.main()


async def test_session_distillation_demo(isolated_example_env):
    module = import_example("examples/demos/session_distillation_demo.py")
    await module.main()


async def test_session_feedback_example(isolated_example_env):
    module = import_example("examples/demos/session_feedback_example.py")
    await module.main()


async def test_session_flow_stepwise_demo(isolated_example_env):
    module = import_example("examples/demos/session_flow_stepwise_demo.py")
    await module.main()


async def test_simple_cognee_example(isolated_example_env):
    module = import_example("examples/demos/simple_cognee_example.py")
    await module.main()


@pytest.mark.xfail(
    reason=(
        "The skill agent expects a JSON answer, but the harness returns the plain "
        "string MOCK_ANSWER ('Agent answer did not contain JSON'). Still fails on "
        "#3958's b5c57d704 hardening (its JSON mock did not reach the agent path); "
        "tracked as a further harness follow-up."
    ),
    strict=False,
)
async def test_skill_feedback_loop_demo(isolated_example_env):
    module = import_example("examples/demos/skill_feedback_loop/skill_feedback_loop_demo.py")
    await module.main()


async def test_temporal_awareness_example(isolated_example_env):
    module = import_example(
        "examples/demos/temporal_awareness_example/temporal_awareness_example.py"
    )
    await module.main()


async def test_truth_centroid_slots_demo(isolated_example_env):
    module = import_example("examples/demos/truth_centroid_slots_demo.py")
    await module.main()


# ---------------------------------------------------------------------------
# Non-uniform entrypoints.
# ---------------------------------------------------------------------------


async def test_hybrid_context_only_demo(isolated_example_env):
    # main() takes two flags; run the full ingest + completion path.
    module = import_example("examples/demos/hybrid_context_only_demo.py")
    await module.main(skip_ingest=False, skip_completion=False)


@pytest.mark.xfail(
    reason=(
        "Example is outdated: it patches "
        "cognee.infrastructure.session.session_manager.analyze_turn_for_session_context, "
        "which no longer exists. The example needs updating against current cognee "
        "(the mocked test correctly caught the rot)."
    ),
    strict=False,
)
async def test_session_context_growth_demo(isolated_example_env):
    # Entrypoint is run_demo(), not main().
    module = import_example("examples/demos/session_context_growth_demo.py")
    await module.run_demo()


async def test_simple_document_qa_demo(isolated_example_env):
    # Entrypoint is cognee_demo(), not main().
    module = import_example("examples/demos/simple_document_qa/simple_document_qa_demo.py")
    await module.cognee_demo()


@pytest.mark.skip(
    reason="pipeline_api_proposal.py is a design-proposal reference, not a runnable/importable example (TypeError at import); no entrypoint to exercise."
)
def test_pipeline_api_proposal_imports(isolated_example_env):
    import_example("examples/demos/pipeline_api_proposal.py")


# ---------------------------------------------------------------------------
# Relational-migration demos: MIGRATION_DB_PROVIDER is read at import time, so
# it must be forced to sqlite before the module body executes.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Relational migration hardcodes MIGRATION_DB_PROVIDER=postgres at import and needs psycopg2 + a source Postgres (Testcontainers); deferred to the database-backends batch."
)
async def test_simple_relational_database_migration_example(isolated_example_env):
    module = import_example(
        "examples/demos/simple_relational_database_migration_example/"
        "simple_relational_database_migration_example.py",
    )
    await module.main()


@pytest.mark.skip(
    reason="Relational migration hardcodes MIGRATION_DB_PROVIDER=postgres at import and needs psycopg2 + a source Postgres (Testcontainers); deferred to the database-backends batch."
)
async def test_complex_relational_database_migration_example(isolated_example_env):
    module = import_example(
        "examples/demos/complex_relational_database_migration_example/"
        "complex_relational_database_migration_example.py",
    )
    await module.main()


# ---------------------------------------------------------------------------
# Genuinely external: not run in the keyless, no-network suite.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Launches the cognee UI server (cognee.start_ui); out of scope for a keyless unit harness."
)
async def test_start_local_ui_frontend_example(isolated_example_env):
    module = import_example("examples/demos/start_local_ui_frontend_example.py")
    await module.main()


@pytest.mark.skip(
    reason="A FastAPI + uvicorn web backend, not a runnable script; needs a TestClient harness (follow-up)."
)
async def test_session_feedback_lifecycle_backend(isolated_example_env):
    import_example("examples/demos/session_feedback_lifecycle_demo/backend/app.py")


@pytest.mark.skip(reason="Fetches a live web URL; not run in the no-network suite.")
async def test_web_url_content_ingestion_example(isolated_example_env):
    module = import_example("examples/demos/web_url_content_ingestion_example.py")
    await module.main()
