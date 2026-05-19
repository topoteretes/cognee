from __future__ import annotations

import argparse
import asyncio
import os


DATASET = "global_context_graph_inputs_demo"

DEMO_TURNS = [
    "Alice and Bruno planned Project Atlas kickoff for April 9.",
    "Bruno said Project Atlas depends on the Graph Inputs provider.",
    "Casey will review the provider diagnostics after the kickoff.",
]


def main() -> None:
    os.environ.setdefault("COGNEE_CLI_MODE", "true")
    os.environ.setdefault("COGNEE_LOG_FILE", "false")
    os.environ.setdefault("LOG_LEVEL", "ERROR")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_id",
        nargs="?",
        default=os.environ.get("COGNEE_DATASET_ID", ""),
        help="Existing dataset UUID. If omitted, the demo ingests a tiny smoke dataset.",
    )
    args = parser.parse_args()

    asyncio.run(_run(args.dataset_id))


async def _run(dataset_id: str) -> None:
    if not dataset_id:
        dataset_id = await _ingest_smoke_dataset()
        if not dataset_id:
            return

    try:
        summary_ids, graph_input = await _load_graph_inputs(dataset_id)
    except ValueError as error:
        print(f"Setup: {error}")
        return

    if not summary_ids:
        print(f"Dataset id: {dataset_id}")
        print("Setup: no TextSummary rows found for this dataset.")
        return

    diagnostics = graph_input.summary_entities
    if diagnostics.missing_made_from_summary_ids:
        missing_count = len(diagnostics.missing_made_from_summary_ids)
        print(f"Dataset id: {dataset_id}")
        print(
            "Setup: "
            f"{missing_count} of {len(summary_ids)} TextSummary rows are missing made_from "
            "graph rows. Run global context graph bucketing on data indexed with pipeline "
            "context so relational graph rows are available."
        )
        return

    print(f"Dataset id: {dataset_id}")
    print(f"TextSummary rows: {len(summary_ids)}")
    print(f"Summarized chunks: {diagnostics.summarized_chunk_count}")
    print(f"Entity links: {diagnostics.entity_link_count}")

    print("Sample summary entities:")
    for summary_id in sorted(summary_ids)[:5]:
        entity_ids = sorted(graph_input.entities_by_summary_id.get(summary_id, set()))
        print(f"  {summary_id}: {entity_ids[:10]}")

    print("Top IDF entities:")
    top_entities = sorted(
        graph_input.idf_weights.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]
    if not top_entities:
        print("  none")
        return

    for entity_id, weight in top_entities:
        print(f"  {entity_id}: {weight:.4f}")


async def _ingest_smoke_dataset() -> str:
    import cognee

    print(f"No dataset id supplied. Ingesting smoke data into dataset: {DATASET}")
    try:
        result = await cognee.remember(
            DEMO_TURNS,
            dataset_name=DATASET,
            self_improvement=False,
        )
    except Exception as error:
        print(
            "Setup: smoke ingestion failed. Ensure LLM configuration is available, "
            f"or pass an existing dataset UUID. Error: {error}"
        )
        return ""

    dataset_id = getattr(result, "dataset_id", None)
    if not dataset_id:
        print("Setup: smoke ingestion completed but did not return a dataset id.")
        return ""

    return str(dataset_id)


async def _load_graph_inputs(dataset_id: str):
    from cognee.modules.graph.methods.get_global_context_graph_inputs import (
        get_dataset_text_summary_ids,
    )
    from cognee.tasks.memify.global_context_index.graph_providers import (
        load_global_context_graph_input,
    )

    summary_ids = await get_dataset_text_summary_ids(dataset_id)
    graph_input = await load_global_context_graph_input(dataset_id, summary_ids)
    return summary_ids, graph_input


if __name__ == "__main__":
    main()
