"""Use durable user decisions even when an obsolete vector row wins retrieval.

Vector rows and source documents remain historical evidence. This overlay changes
only retrieved copies, and never relies on a later re-embedding or improve run.
"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import copy

from cognee.context_global_variables import current_dataset_id
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.fact_conflicts import read_conflict_marks, read_list_property
from cognee.modules.retrieval.utils.results import payload, result_id

_resolution_reads: ContextVar[dict | None] = ContextVar("resolution_reads", default=None)


@asynccontextmanager
async def user_resolution_scope():
    """Share reads within one search, including concurrent lanes and follow-up queries."""
    if _resolution_reads.get() is not None:
        yield
        return
    reads = {}
    token = _resolution_reads.set(reads)
    try:
        yield
    finally:
        _resolution_reads.reset(token)
        pending = [task for task in reads.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def read_user_resolutions(graph) -> list[dict]:
    reads = _resolution_reads.get()
    if reads is None:
        return await _read_user_resolutions(graph)
    key = (id(graph), current_dataset_id.get())
    if key not in reads:
        reads[key] = asyncio.create_task(_read_user_resolutions(graph))
    return await reads[key]


async def _read_user_resolutions(graph) -> list[dict]:
    nodes, _ = await graph.get_filtered_graph_data(
        [{"type": ["FactConflict"]}], include_edges=False
    )
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


def _fact_id(edge) -> str:
    return str(
        edge.attributes.get("edge_object_id")
        or generate_edge_object_id(
            edge.node1.id,
            edge.node2.id,
            edge.attributes.get("relationship_name")
            or edge.attributes.get("relationship_type")
            or "",
        )
    )


def apply_graph_resolutions(edges: list, resolutions: list[dict]) -> list:
    """Replace only explicitly rejected edges, preserving all other retrieved content.

    Each decision is stated once in these edges, read from the decision node: a document
    rebuild re-creates its edges without their marks. A retrieved selected edge is
    labelled as the user's choice, and the rejected edges retrieved with it are dropped.
    Otherwise the decision's first rejected edge becomes one synthetic link carrying
    the accepted fact, and its other rejected edges are dropped.
    """
    if not resolutions:
        return edges
    rejected = {
        fid: resolution for resolution in resolutions for fid in resolution["rejected_fact_ids"]
    }
    selected = {
        str(resolution["selected_fact_id"]): resolution
        for resolution in resolutions
        if resolution.get("selected_fact_id")
    }
    fact_ids = [_fact_id(edge) for edge in edges]
    retrieved = {fact_id for fact_id in fact_ids if fact_id not in rejected}
    linked: set[str] = set()
    result = []
    for edge, fact_id in zip(edges, fact_ids):
        updated = copy(edge)
        updated.attributes = dict(edge.attributes)
        resolution = rejected.get(fact_id)
        if resolution:
            if str(resolution.get("selected_fact_id")) in retrieved or resolution["id"] in linked:
                continue
            linked.add(resolution["id"])
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
            # Every fact of a conflict touches its subject. Linking to that node as
            # retrieved keeps its name and description on the subject's other edges.
            subject_first = str(edge.node1.id) == str(resolution["about_id"])
            updated.node1, updated.node2 = source, edge.node1 if subject_first else edge.node2
        elif fact_id in selected:
            decision = selected[fact_id]
            updated.attributes["conflict_marks"] = [
                *(
                    mark
                    for mark in read_conflict_marks(edge.attributes)
                    if str(mark.get("conflict_id")) != decision["id"]
                ),
                {
                    "conflict_id": decision["id"],
                    "status": "current",
                    "resolution_source": "user",
                    "dataset_id": decision.get("dataset_id"),
                    "selected_fact": decision["selected_fact"],
                },
            ]
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


def accepted_facts_context(
    resolutions: list[dict], node_ids: list[str], edge_ids: list[str]
) -> str:
    """Every user decision about the retrieved nodes or their sources, for the prelude.

    A decision whose selected fact is in the context is left out: that edge, or the
    synthetic link standing in for it, already states the decision.
    """
    retrieved_ids = set(node_ids)
    stated = set(edge_ids)
    accepted = sorted(
        {
            user_resolution_text(item["selected_fact"])
            for item in resolutions
            if str(item.get("selected_fact_id")) not in stated
            and retrieved_ids.intersection(
                str(node_id)
                for node_id in (item.get("about_id"), *item["values"], *item["sources"])
                if node_id is not None
            )
        }
    )
    if not accepted:
        return ""
    return "## Accepted facts (user resolved)\n" + "\n".join(accepted)
