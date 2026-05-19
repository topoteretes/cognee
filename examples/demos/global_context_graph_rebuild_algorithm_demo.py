from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("COGNEE_CLI_MODE", "true")
    os.environ.setdefault("COGNEE_LOG_FILE", "false")
    os.environ.setdefault("LOG_LEVEL", "ERROR")

    from cognee.tasks.memify.global_context_index.graph_bucketing import (
        rebuild_graph_buckets_for_level,
    )
    from cognee.tasks.memify.global_context_index.models import SummaryNode

    summaries = [
        SummaryNode(id="summary-a", text="Alice met about Project X.", type="TextSummary"),
        SummaryNode(id="summary-b", text="Alice updated the roadmap.", type="TextSummary"),
        SummaryNode(id="summary-c", text="Bob reviewed Project Y.", type="TextSummary"),
        SummaryNode(id="summary-d", text="Unstructured follow-up note.", type="TextSummary"),
        SummaryNode(id="summary-e", text="Daily standup only.", type="TextSummary"),
    ]
    entities_by_summary_id = {
        "summary-a": {"alice", "project-x", "standup"},
        "summary-b": {"alice", "roadmap", "standup"},
        "summary-c": {"bob", "project-y", "standup"},
        "summary-d": set(),
        "summary-e": {"standup"},
    }
    idf_weights = {
        "alice": 1.0,
        "project-x": 1.4,
        "roadmap": 1.4,
        "bob": 1.4,
        "project-y": 1.4,
        "standup": 0.0,
    }

    buckets, assignments = rebuild_graph_buckets_for_level(
        summaries,
        entities_by_summary_id,
        idf_weights,
        dataset_id="demo-dataset",
        level=0,
        max_bucket_size=2,
        min_overlap=0.1,
    )

    print("Global context graph rebuild algorithm demo")
    for bucket in buckets.values():
        child_ids = sorted(bucket.child_ids)
        entity_ids = sorted(bucket.graph_bucket_entity_ids or set())
        print(f"bucket: {bucket.id}")
        print(f"  children: {child_ids}")
        print(f"  graph_bucket_entity_ids: {entity_ids}")

    print("assignments:")
    for assignment in assignments:
        print(f"  {assignment.summary_id} -> {assignment.bucket_id}")


if __name__ == "__main__":
    main()
