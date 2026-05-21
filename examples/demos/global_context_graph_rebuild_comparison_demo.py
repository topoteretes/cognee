from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from uuid import UUID


DATASET = "global_context_graph_rebuild_demo"

DEMO_TURNS = [
    "Alice and Bruno planned Project Atlas kickoff for April 9.",
    "Bruno said Project Atlas depends on the graph inputs provider.",
    "Casey will review Project Atlas diagnostics after the kickoff.",
    "Dina and Eli planned Project Beacon onboarding for May 3.",
    "Eli said Project Beacon depends on the search retriever updates.",
    "Farah will review Project Beacon rollout notes after onboarding.",
]

INCREMENTAL_TURNS = [
    "Alice added a Project Atlas follow-up focused on provider validation.",
    "Eli added a Project Beacon follow-up focused on retriever rollout metrics.",
]


def main() -> None:
    os.environ.setdefault("COGNEE_CLI_MODE", "true")
    os.environ.setdefault("COGNEE_LOG_FILE", "false")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    if not asyncio.run(_run()):
        sys.exit(1)


async def _run() -> bool:
    print("Global context vector rebuild, graph rebuild, and graph incremental demo")

    dataset_id = await _ingest_dataset(DATASET)
    if not dataset_id:
        return False

    try:
        await _build_index(dataset_id, strategy="vector", rebuild=True)
        if await _print_buckets("Vector rebuild buckets", dataset_id) is None:
            return False

        print("\nGraph rebuild will replace the vector index for the same dataset.")
        await _build_index(dataset_id, strategy="graph", rebuild=True)
        graph_bucket_ids = await _print_buckets("Graph rebuild buckets", dataset_id)
        if graph_bucket_ids is None:
            return False

        await _ingest_more_data(dataset_id)
        await _build_index(dataset_id, strategy="graph", rebuild=False)
        incremental_bucket_ids = await _print_buckets("Graph incremental buckets", dataset_id)
        if incremental_bucket_ids is None:
            return False

        print("\nGraph incremental bucket changes")
        print(f"  reused bucket ids: {sorted(graph_bucket_ids & incremental_bucket_ids)}")
        print(f"  new bucket ids: {sorted(incremental_bucket_ids - graph_bucket_ids)}")
    except Exception as error:
        print(
            "Setup: demo failed. Ensure LLM configuration is available and the "
            f"relational graph rows are populated. Error: {error}"
        )
        return False

    return True


async def _ingest_dataset(dataset_name: str) -> str:
    import cognee

    print(f"Ingesting dataset: {dataset_name}")
    try:
        result = await cognee.remember(
            DEMO_TURNS,
            dataset_name=dataset_name,
            self_improvement=False,
        )
    except Exception as error:
        print(f"Setup: ingestion failed. Ensure LLM configuration is available. Error: {error}")
        return ""

    dataset_id = getattr(result, "dataset_id", None)
    if not dataset_id:
        print("Setup: ingestion completed but did not return a dataset id.")
        return ""
    return str(dataset_id)


async def _ingest_more_data(dataset_id: str) -> None:
    import cognee

    print("\nIngesting additional data for graph incremental update")
    await cognee.remember(
        INCREMENTAL_TURNS,
        dataset_name=DATASET,
        dataset_id=UUID(dataset_id),
        self_improvement=False,
    )


async def _build_index(dataset_id: str, strategy: str, rebuild: bool) -> None:
    from cognee.memify_pipelines.global_context_index import global_context_index_pipeline
    from cognee.modules.users.methods import get_default_user

    mode = "rebuild" if rebuild else "incremental update"
    print(f"\nRunning {strategy} {mode} on dataset id: {dataset_id}")
    user = await get_default_user()
    result = await global_context_index_pipeline(
        user=user,
        dataset=UUID(dataset_id),
        rebuild=rebuild,
        bucketing_strategy=strategy,
        max_bucket_size=2,
    )
    print(f"{strategy} {mode} result: {result}")


async def _print_buckets(title: str, dataset_id: str) -> set[str] | None:
    from cognee.modules.graph.methods.get_global_context_graph_inputs import (
        get_dataset_text_summary_ids,
    )
    from cognee.modules.pipelines.models import PipelineContext
    from cognee.tasks.memify.global_context_index.graph_input import (
        load_context_index_input_from_graph,
    )
    from cognee.tasks.memify.global_context_index.graph_providers import (
        load_global_context_graph_input,
    )

    summary_ids = await get_dataset_text_summary_ids(dataset_id)
    graph_input = await load_global_context_graph_input(dataset_id, summary_ids)
    context_input = await load_context_index_input_from_graph(
        PipelineContext(dataset=SimpleNamespace(id=dataset_id))
    )

    print(f"\n{title}")
    print(f"Dataset id: {dataset_id}")
    print(f"TextSummary rows: {len(summary_ids)}")
    print(f"Graph TextSummary nodes loaded: {len(context_input.text_summaries)}")
    print(f"GlobalContextSummary nodes loaded: {len(context_input.buckets)}")
    print(f"Root loaded: {context_input.root is not None}")

    summaries_by_id = {summary.id: summary for summary in context_input.text_summaries}
    level_zero_buckets = sorted(
        [bucket for bucket in context_input.buckets if bucket.level == 0],
        key=lambda bucket: bucket.id,
    )
    if not level_zero_buckets:
        print("  no level-0 buckets; rebuild did not produce a usable index")
        return None

    for bucket in level_zero_buckets:
        child_ids = sorted(bucket.child_ids)
        top_entities = _top_bucket_entities(child_ids, bucket.graph_bucket_entity_ids, graph_input)
        print(f"  bucket {bucket.id}")
        print(f"    size: {len(child_ids)}")
        print(f"    child summary ids: {child_ids}")
        print(f"    child previews: {_child_previews(child_ids, summaries_by_id)}")
        print(f"    top entities: {top_entities}")

    return {bucket.id for bucket in level_zero_buckets}


def _top_bucket_entities(child_ids, graph_bucket_entity_ids, graph_input):
    entity_ids = set(graph_bucket_entity_ids or [])
    if graph_bucket_entity_ids is None:
        for child_id in child_ids:
            entity_ids.update(graph_input.entities_by_summary_id.get(child_id, set()))

    weighted_entities = [
        (entity_id, graph_input.idf_weights.get(entity_id, 0.0)) for entity_id in entity_ids
    ]
    return sorted(weighted_entities, key=lambda item: (-item[1], item[0]))[:5]


def _child_previews(child_ids, summaries_by_id) -> list[str]:
    previews = []
    for child_id in child_ids:
        summary = summaries_by_id.get(child_id)
        text = summary.text if summary else ""
        previews.append(text[:80])
    return previews


if __name__ == "__main__":
    main()
