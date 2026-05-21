from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("COGNEE_CLI_MODE", "true")
    os.environ.setdefault("COGNEE_LOG_FILE", "false")
    os.environ.setdefault("LOG_LEVEL", "ERROR")

    from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
    from cognee.tasks.memify.global_context_index.graph_input import (
        extract_context_index_input_from_graph,
    )

    dataset_id = "demo-dataset"
    graph = CogneeGraph()
    for node in [
        _bucket_node("missing-state", dataset_id, include_entity_state=False),
        _bucket_node("null-state", dataset_id, graph_bucket_entity_ids=None),
        _bucket_node("empty-state", dataset_id, graph_bucket_entity_ids=[]),
        _bucket_node(
            "populated-state",
            dataset_id,
            graph_bucket_entity_ids=["entity-b", "entity-a"],
        ),
    ]:
        graph.add_node(node)

    context_input = extract_context_index_input_from_graph(graph, dataset_id)

    print("Global context graph bucket entity state roundtrip demo")
    for bucket in sorted(context_input.buckets, key=lambda bucket: bucket.id):
        entity_state = bucket.graph_bucket_entity_ids
        rendered_state = None if entity_state is None else sorted(entity_state)
        print(f"{bucket.id}: graph_bucket_entity_ids={rendered_state}")


def _bucket_node(
    node_id: str,
    dataset_id: str,
    *,
    graph_bucket_entity_ids: list[str] | None = None,
    include_entity_state: bool = True,
):
    from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node

    attributes = {
        "type": "GlobalContextSummary",
        "text": node_id,
        "dataset_id": dataset_id,
        "level": 0,
        "is_root": False,
    }
    if include_entity_state:
        attributes["graph_bucket_entity_ids"] = graph_bucket_entity_ids
    return Node(node_id, attributes)


if __name__ == "__main__":
    main()
