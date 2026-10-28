"""Use durable user decisions even when an obsolete vector row wins retrieval.

Vector rows and source documents remain historical evidence. This overlay changes
only retrieved copies, and never relies on a later re-embedding or improve run.
"""

from copy import copy

from cognee.context_global_variables import current_dataset_id
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.fact_conflicts import read_list_property
from cognee.modules.retrieval.utils.results import payload, result_id


async def read_user_resolutions(graph) -> list[dict]:
    nodes, _ = await graph.get_filtered_graph_data([{"type": ["FactConflict"]}])
    dataset_id = current_dataset_id.get()
    resolutions = []
    for node_id, props in nodes:
        if props.get("resolution_source") != "user" or not props.get("selected_fact"):
            continue
        if dataset_id is not None and str(props.get("dataset_id")) != str(dataset_id):
            continue
        resolutions.append(
            {
                **props,
                "id": str(node_id),
                **{
                    name: read_list_property(props, name)
                    for name in ("sources", "values", "rejected_fact_ids", "rejected_fact_texts")
                },
            }
        )
    return resolutions


def apply_graph_resolutions(edges: list, resolutions: list[dict]) -> list:
    """Replace rejected claims with the selected claim and sanitize cited source text."""
    if not resolutions:
        return edges
    rejected = {
        fid: resolution for resolution in resolutions for fid in resolution["rejected_fact_ids"]
    }
    by_chunk: dict[str, set[str]] = {}
    by_entity: dict[str, set[str]] = {}
    for resolution in resolutions:
        for chunk_id in resolution["sources"]:
            by_chunk.setdefault(chunk_id, set()).add(resolution["selected_fact"])
        for node_id in (resolution["about_id"], *resolution.get("values", [])):
            by_entity.setdefault(str(node_id), set()).add(resolution["selected_fact"])
    result = []
    for edge in edges:
        updated = copy(edge)
        updated.attributes = dict(edge.attributes)
        fact_id = edge.attributes.get("edge_object_id") or generate_edge_object_id(
            edge.node1.id,
            edge.node2.id,
            edge.attributes.get("relationship_name")
            or edge.attributes.get("relationship_type")
            or "",
        )
        resolution = rejected.get(str(fact_id))
        if resolution:
            # A context-only link from the real resolution node to its subject carries
            # the accepted assertion. Evidence identifies the selected fact, never the
            # rejected edge that happened to win vector retrieval.
            updated.attributes.update(
                edge_text=resolution["selected_fact"],
                edge_object_id=resolution["selected_fact_id"],
                relationship_name="accepted_fact",
                relationship_type="accepted_fact",
                conflict_marks=[],
            )
            source = copy(edge.node1)
            source.id = resolution["id"]
            source.attributes = {"name": "Accepted fact", "text": resolution["selected_fact"]}
            target = copy(edge.node2)
            target.id = resolution["about_id"]
            target.attributes = {"name": "Subject of the user resolution"}
            updated.node1, updated.node2 = source, target
        else:
            source_facts: set[str] = set()
            for field in ("node1", "node2"):
                node = getattr(edge, field)
                chunk_id = str(node.attributes.get("source_chunk_id") or node.id)
                if chunk_id in by_chunk:
                    source_facts.update(by_chunk[chunk_id])
                    node = copy(node)
                    node.attributes = {
                        **node.attributes,
                        "text": "\n".join(sorted(by_chunk[chunk_id])),
                    }
                    setattr(updated, field, node)
                elif str(node.id) in by_entity:
                    node = copy(node)
                    node.attributes = {
                        **node.attributes,
                        "description": "\n".join(sorted(by_entity[str(node.id)])),
                    }
                    setattr(updated, field, node)
            if source_facts:
                updated.attributes["edge_text"] = "\n".join(sorted(source_facts))
        result.append(updated)
    return result


def apply_fact_hit_resolutions(hits: list, resolutions: list[dict]) -> list:
    """Replace standalone obsolete EdgeType hits, which otherwise bypass graph marks."""
    replacements: dict[str, set[str]] = {}
    for resolution in resolutions:
        for text in resolution["rejected_fact_texts"]:
            replacements.setdefault(str(EdgeType.id_for(text)), set()).add(
                resolution["selected_fact"]
            )
    updated = []
    for hit in hits:
        texts = replacements.get(result_id(hit))
        if not texts:
            updated.append(hit)
            continue
        text = "\n".join(sorted(texts))
        props = {**payload(hit), "text": text, "relationship_name": text}
        if isinstance(hit, dict):
            updated.append(props)
        else:
            item = copy(hit)
            item.payload = props
            updated.append(item)
    return updated


def apply_entity_resolutions(entities: list[dict], resolutions: list[dict]) -> None:
    """Attach accepted facts even if the selected edge fell outside the top-k window."""
    by_chunk: dict[str, set[str]] = {}
    for item in resolutions:
        for chunk_id in item.get("sources", []):
            by_chunk.setdefault(chunk_id, set()).add(item["selected_fact"])
    for entity in entities:
        for edge in entity.get("edges", []):
            source_facts = by_chunk.get(str(edge.get("source_id")), set()) | by_chunk.get(
                str(edge.get("target_id")), set()
            )
            if source_facts:
                edge["text"] = "\n".join(sorted(source_facts))
        relevant = [
            item
            for item in resolutions
            if str(entity.get("id")) in {str(item["about_id"]), *item.get("values", [])}
        ]
        if not relevant:
            continue
        accepted = sorted({item["selected_fact"] for item in relevant})
        entity["conflicts"] = sorted(set(entity.get("conflicts", [])) | set(accepted))
        rejected = {fid for item in relevant for fid in item["rejected_fact_ids"]}
        entity["edges"] = [
            edge
            for edge in entity.get("edges", [])
            if str(edge.get("edge_object_id")) not in rejected
        ]


def accepted_facts_context(resolutions: list[dict]) -> str:
    if not resolutions:
        return ""
    return "## Accepted facts (user resolved)\n" + "\n".join(
        sorted({item["selected_fact"] for item in resolutions})
    )
