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
    """Replace only explicitly rejected edges, preserving all other retrieved content."""
    if not resolutions:
        return edges
    rejected = {
        fid: resolution for resolution in resolutions for fid in resolution["rejected_fact_ids"]
    }
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
    for entity in entities:
        relevant = [
            item
            for item in resolutions
            if str(entity.get("id")) in {str(item["about_id"]), *item.get("values", [])}
        ]
        if not relevant:
            continue
        accepted = {user_resolution_text(item["selected_fact"]) for item in relevant}
        # Neighborhood marks and conflict nodes may already carry the same decision.
        existing_decisions = {
            text for item in relevant for text in (item["selected_fact"], item.get("text")) if text
        }
        entity["conflicts"] = sorted(
            (set(entity.get("conflicts", [])) - existing_decisions) | accepted
        )
        rejected = {fid for item in relevant for fid in item["rejected_fact_ids"]}
        entity["edges"] = [
            edge
            for edge in entity.get("edges", [])
            if str(edge.get("edge_object_id")) not in rejected
        ]


def user_resolution_text(selected_fact: str) -> str:
    """Give a stored user decision precedence over competing source claims."""
    return f"Accepted by user: {selected_fact}; ignore competing claims."


def accepted_facts_context(resolutions: list[dict], node_ids: list[str], limit: int) -> str:
    """Bound the optional prelude to decisions about the retrieved nodes and sources."""
    if limit <= 0:
        return ""
    retrieved_ids = set(node_ids)
    accepted = sorted(
        {
            user_resolution_text(item["selected_fact"])
            for item in resolutions
            if retrieved_ids.intersection(
                str(node_id)
                for node_id in (item.get("about_id"), *item["values"], *item["sources"])
                if node_id is not None
            )
        }
    )[:limit]
    if not accepted:
        return ""
    return "## Accepted facts (user resolved)\n" + "\n".join(accepted)
