"""Attach the stored conflicts to a search's hits, and render them for a prompt.

Two jobs, in that order. ``attach_chunk_conflicts`` annotates the hits a
retriever just selected; the renderers below turn annotated hits into the
passage and conflict sections of a retrieval context. Both are additive: a
failure leaves the hits exactly as they were retrieved rather than failing the
search, which is why the graph read is guarded here rather than at each caller.
"""

from dataclasses import dataclass, field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils.fact_conflicts import CONFLICT_CITES, effective_date_display
from cognee.modules.retrieval.utils.results import payload, result_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("conflict_context")

# The header rides inside the hit's payload rather than in a side table because
# merge_ranked (the two session retrieval lanes) and merge_hybrid_results move
# payload objects between result sets and carry nothing beside them — a header
# held outside the payload would be dropped by the very merge that picks the
# surviving hit. public_chunk_payload strips it again before a payload is handed
# back to a caller.
PASSAGE_HEADER_KEY = "_passage_header"


@dataclass
class ChunkConflicts:
    """What one chunk's citations say: its source document, and conflict id -> explanation."""

    document: str | None = None
    effective_date: str | None = None
    conflicts: dict[str, str] = field(default_factory=dict)


def _unwrap_hit(item):
    """The hit itself; lexical search hands back ``(payload, score)`` pairs."""
    return item[0] if isinstance(item, tuple) else item


def _hit_payload(item) -> dict:
    return payload(_unwrap_hit(item))


# --- reading the stored conflicts -------------------------------------------


def _citations_by_chunk(nodes, edges, *, requested_ids: set[str]) -> dict[str, ChunkConflicts]:
    """Group the conflicts citing each requested chunk; the first citation sets the header."""
    conflict_properties = {str(node_id): properties for node_id, properties in nodes}
    citations = sorted(
        (
            (str(target), str(source), edge_properties)
            # Neighborhood filters select neighbors, not the returned induced edges.
            for source, target, relationship, edge_properties in edges
            if relationship == CONFLICT_CITES and str(target) in requested_ids
        ),
        key=lambda citation: citation[:2],
    )
    by_chunk: dict[str, ChunkConflicts] = {}
    for chunk_id, conflict_id, edge_properties in citations:
        text = conflict_properties.get(conflict_id, {}).get("text") or edge_properties.get(
            "edge_text"
        )
        if not text:
            continue
        chunk = by_chunk.setdefault(
            chunk_id,
            ChunkConflicts(edge_properties.get("document"), edge_properties.get("effective_date")),
        )
        chunk.conflicts[conflict_id] = text
    return by_chunk


async def get_chunk_conflicts(graph_engine, chunk_ids: list[str]) -> dict[str, ChunkConflicts]:
    """Read the conflicts citing each selected chunk, across all datasets in scope.

    Unguarded on purpose. The obvious guard — the review watermark — is a stamp on a
    succeeded improve row, so it reads absent whenever a later stage errored, a write
    failed mid-run, or ``review_conflicts_pipeline`` was driven directly, each of which
    leaves stored conflicts the guard would silently hide. Asking the graph for a
    ``FactConflict`` instead costs the unindexed type scan the write path pays. The id
    check in ``_citations_by_chunk`` is the only guard that cannot go stale.
    """
    if not chunk_ids:
        return {}
    try:
        nodes, edges = await graph_engine.get_neighborhood(
            chunk_ids, depth=1, edge_types=[CONFLICT_CITES]
        )
        return _citations_by_chunk(nodes, edges, requested_ids=set(chunk_ids))
    except Exception as error:
        logger.warning("Unable to load chunk conflicts: %s", error, exc_info=True)
        return {}


# --- annotating a search's hits ---------------------------------------------


def _payloads_by_chunk_id(hits, *, summaries: bool) -> dict[str, list[dict]]:
    """Hit payloads to annotate, keyed by the chunk whose conflicts cover them.

    A summary is not a chunk: it names its chunk in ``source_chunk_id``, and
    several summaries can share one chunk, so each key holds a list.
    """
    payloads_by_chunk_id: dict[str, list[dict]] = {}
    for item in hits or []:
        hit = _unwrap_hit(item)
        hit_payload = payload(hit)
        chunk_id = hit_payload.get("source_chunk_id") if summaries else result_id(hit)
        if chunk_id:
            payloads_by_chunk_id.setdefault(str(chunk_id), []).append(hit_payload)
    return payloads_by_chunk_id


async def attach_chunk_conflicts(hits, *, graph_engine=None, summaries: bool = False) -> None:
    """Annotate the selected hits in place with the conflicts citing them.

    Total by design, so a caller never needs its own guard: resolving the graph
    engine can fail on its own, which ``get_chunk_conflicts`` never sees.
    Callers that already hold an engine pass it.
    """
    payloads_by_chunk_id = _payloads_by_chunk_id(hits, summaries=summaries)
    if not payloads_by_chunk_id:
        return
    try:
        engine = graph_engine or await get_graph_engine()
        cited_by_chunk_id = await get_chunk_conflicts(engine, list(payloads_by_chunk_id))
    except Exception:
        logger.warning("Could not read chunk conflicts", exc_info=True)
        return
    for chunk_id, cited in cited_by_chunk_id.items():
        explanations = [cited.conflicts[key] for key in sorted(cited.conflicts)]
        header = passage_header(cited.document, cited.effective_date)
        for hit_payload in payloads_by_chunk_id[chunk_id]:
            hit_payload["conflicts"] = list(explanations)
            hit_payload[PASSAGE_HEADER_KEY] = header


def public_chunk_payload(hit) -> dict:
    """Copy a hit's payload without the keys only the context renderers use."""
    return {key: value for key, value in _hit_payload(hit).items() if key != PASSAGE_HEADER_KEY}


# --- rendering a retrieval context ------------------------------------------


def passage_header(document: str | None, effective_date) -> str:
    """Name the document a passage came from and, where known, the date it states."""
    if not document:
        return ""
    date = effective_date_display(effective_date)
    return f"{document} ({date})" if date else document


def join_passage_texts(payloads) -> str:
    """Each passage's text, preceded by its source header where one was attached."""
    passages = []
    for item in payloads or []:
        chunk = _hit_payload(item)
        source = chunk.get(PASSAGE_HEADER_KEY)
        passages.append(f"source: {source}\n{chunk['text']}" if source else chunk["text"])
    return "\n".join(passages)


def format_conflicts(texts) -> str:
    """One deduplicated block of conflict explanations, or "" when there are none."""
    unique_texts = sorted({text for text in texts if text})
    if not unique_texts:
        return ""
    return "## Fact conflicts\n" + "\n".join(f"- {text}" for text in unique_texts)


def render_chunk_context(payloads) -> str:
    """Render source passages followed by their deduplicated conflict explanations."""
    passages = join_passage_texts(payloads)
    conflicts = format_conflicts(
        text for item in payloads for text in _hit_payload(item).get("conflicts", [])
    )
    return "\n\n".join(part for part in (passages, conflicts) if part)
