from __future__ import annotations

from typing import Any

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment

from .constants import SUMMARIZED_IN, SUMMARY_GRAPH_NODE_TYPES
from .models import GlobalContextIndexUpdateData, SummaryNode


def dataset_id_from_context(ctx: PipelineContext | None) -> str:
    dataset = ctx.dataset if ctx else None
    dataset_id = getattr(dataset, "id", dataset)
    return str(dataset_id) if dataset_id is not None else ""


def global_context_summary_level(attributes: dict[str, Any]) -> int | None:
    try:
        return int(attributes.get("level"))
    except (TypeError, ValueError):
        return None


def is_root_global_context_summary(attributes: dict[str, Any]) -> bool:
    is_root = attributes.get("is_root")
    if isinstance(is_root, bool):
        return is_root
    if isinstance(is_root, str):
        return is_root.lower() == "true"
    return bool(is_root)


def graph_bucket_entity_ids_from_attributes(
    attributes: dict[str, Any],
    level: int | None,
) -> set[str] | None:
    if level != 0:
        return None

    raw_entity_ids = attributes.get("graph_bucket_entity_ids")
    if raw_entity_ids is None:
        return None
    if isinstance(raw_entity_ids, (list, set, tuple)):
        return {str(entity_id) for entity_id in raw_entity_ids}
    return None


def extract_context_index_input(
    memory_fragment: CogneeGraph,
    dataset_id: str,
) -> GlobalContextIndexUpdateData:
    text_summaries: list[SummaryNode] = []
    buckets: list[SummaryNode] = []
    root: SummaryNode | None = None

    for node in memory_fragment.nodes.values():
        attributes = node.attributes
        node_type = attributes.get("type")

        if node_type == "TextSummary":
            text_summaries.append(
                SummaryNode(
                    id=str(node.id),
                    text=str(attributes.get("text") or ""),
                    type="TextSummary",
                )
            )
            continue

        if node_type != "GlobalContextSummary":
            continue

        node_dataset_id = attributes.get("dataset_id")
        if node_dataset_id is not None and str(node_dataset_id) != dataset_id:
            continue

        level = global_context_summary_level(attributes)
        bucket_node = SummaryNode(
            id=str(node.id),
            text=str(attributes.get("text") or ""),
            type="GlobalContextSummary",
            level=level,
            is_root=is_root_global_context_summary(attributes),
            dataset_id=str(node_dataset_id) if node_dataset_id is not None else dataset_id,
            graph_bucket_entity_ids=graph_bucket_entity_ids_from_attributes(attributes, level),
        )
        if bucket_node.is_root:
            root = bucket_node
        else:
            buckets.append(bucket_node)

    nodes_by_id = {
        node.id: node for node in text_summaries + buckets + ([root] if root is not None else [])
    }
    for edge in memory_fragment.edges:
        if relationship_name(edge) != SUMMARIZED_IN:
            continue
        child = nodes_by_id.get(str(edge.node1.id))
        parent = nodes_by_id.get(str(edge.node2.id))
        if child is None or parent is None or child.global_context_bucket_id:
            continue
        if is_valid_context_parent(child, parent):
            child.global_context_bucket_id = parent.id

    rebuild_child_ids_from_parent_pointers(
        text_summaries + buckets,
        buckets + ([root] if root is not None else []),
    )
    return GlobalContextIndexUpdateData(text_summaries=text_summaries, buckets=buckets, root=root)


def relationship_name(edge) -> str | None:
    return edge.attributes.get("relationship_name") or edge.attributes.get("relationship_type")


def is_valid_context_parent(child: SummaryNode, parent: SummaryNode) -> bool:
    if child.id == parent.id or child.is_root:
        return False
    if parent.type != "GlobalContextSummary":
        return False

    if child.type == "TextSummary":
        return not parent.is_root

    if child.type != "GlobalContextSummary":
        return False
    if parent.is_root:
        return True
    if child.level is None or parent.level is None:
        return False
    return parent.level == child.level + 1


def rebuild_child_ids_from_parent_pointers(
    children: list[SummaryNode],
    parents: list[SummaryNode],
) -> None:
    parents_by_id = {parent.id: parent for parent in parents}
    for parent in parents:
        parent.child_ids.clear()

    for child in children:
        if not child.global_context_bucket_id:
            continue
        parent = parents_by_id.get(child.global_context_bucket_id)
        if parent is not None:
            parent.child_ids.add(child.id)


async def load_context_index_input_from_graph(
    ctx: PipelineContext | None,
) -> GlobalContextIndexUpdateData:
    dataset_id = dataset_id_from_context(ctx)
    memory_fragment = await get_context_index_memory_fragment()
    return extract_context_index_input(memory_fragment, dataset_id)


async def get_context_index_memory_fragment() -> CogneeGraph:
    return await get_memory_fragment(
        properties_to_project=[
            "id",
            "text",
            "type",
            "dataset_id",
            "level",
            "is_root",
            "graph_bucket_entity_ids",
            "importance_weight",
            "global_context_bucket_id",
        ],
        memory_fragment_filter=[{"type": SUMMARY_GRAPH_NODE_TYPES}],
    )


async def extract_global_context_index_input(
    data: Any = None,
    ctx: PipelineContext | None = None,
) -> GlobalContextIndexUpdateData:
    if data is not None and data != {} and data != [{}]:
        raise TypeError("extract_global_context_index_input expected None, {}, or [{}].")
    dataset_id = dataset_id_from_context(ctx)
    memory_fragment = await get_context_index_memory_fragment()
    return extract_context_index_input(memory_fragment, dataset_id)
