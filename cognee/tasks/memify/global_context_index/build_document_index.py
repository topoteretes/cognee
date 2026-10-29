"""Task 2 of the ``from_entities`` method: compute the index and write it back.

A bucket holds the entities its document is distinctively about (tf-idf over the
dataset's chunks) and the relations among them; the root holds the dataset's most
widespread names (document frequency). The two rankings are opposite on purpose:
a bucket leads with what sets its document apart, the root with what the dataset
shares.

No LLM call anywhere on this path, which is what makes it cheap to keep current:
every run computes the full desired index, writes the nodes whose text changed,
and deletes the index nodes it did not compute. That one rule covers deleted
documents, incremental additions and switching over from ``method="llm"``, so
there is no placement and no rebuild flag. ``improve()`` skips the run while only
reads have touched the dataset since the last one.

``build_context_summaries`` and its helpers are pure: they take a
``DatasetContent`` and return data.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.graph.utils.entity_scoring import compute_idf_from_counts
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.modules.search.dataset_routing import sync_dataset_routes
from cognee.tasks.summarization.models import GlobalContextSummary

from .constants import (
    DOCUMENT_BUCKET_MAX_ENTITIES,
    DOCUMENT_BUCKET_MAX_RELATIONS,
    ROOT_MAX_DOCUMENTS,
    ROOT_MAX_ENTITIES,
    ROOT_MAX_TYPES,
    SUMMARIZED_IN,
    SUMMARIZES,
)
from .ids import create_document_bucket_id, create_root_summary_id
from .models import DatasetContent
from .persist import (
    delete_context_index_nodes,
    persist_context_index_edges,
    persist_context_summaries,
    safe_delete_context_vectors,
)

_RankedEntities = list[tuple[str, float]]


@task_summary("Updated {n} global context index node(s)")
async def build_document_index(
    data: Any,
    ctx: PipelineContext | None = None,
    link_documents: bool = True,
) -> list[GlobalContextSummary]:
    """Memify enrichment task: write the computed index, drop everything else.

    ``link_documents`` writes the ``summarizes`` edge from each bucket to its
    document. Nothing reads it -- ``document_id`` on the bucket already carries
    the link -- so it is there to keep the graph connected when it is looked at.
    Turning it off only stops new writes; edges already in the graph stay.
    """
    if not isinstance(data, DatasetContent):
        raise TypeError("build_document_index expected a DatasetContent.")

    desired = {str(summary.id): summary for summary in build_context_summaries(data)}
    unified_engine = await get_unified_engine()

    # Deleted documents, a dataset emptied to nothing, and switching over from
    # method="llm" are all the same case, which is why there is no rebuild flag.
    await delete_context_index_nodes(
        unified_engine,
        [node_id for node_id in data.existing_summary_texts if node_id not in desired],
    )

    changed = [
        summary
        for node_id, summary in desired.items()
        if data.existing_summary_texts.get(node_id) != summary.text
    ]
    # A changed summary keeps its id (derived from the dataset and document, not
    # the text), and the vector stores keep a conflicting row's stored vector
    # (their ids normally derive from content). Its old vector goes first, so
    # the write below embeds the new text instead of leaving a stale vector.
    await safe_delete_context_vectors(
        unified_engine.vector,
        [str(summary.id) for summary in changed if str(summary.id) in data.existing_summary_texts],
    )
    await persist_context_summaries(changed, ctx)
    # The routing collection mirrors this index and reconciles against the full
    # desired set, not the diff, so a row it is missing is written too. Its rows
    # copy the vectors just written to this dataset's store.
    await sync_dataset_routes(data.dataset_id, list(desired.values()), unified_engine.vector)
    changed_buckets = [summary for summary in changed if not summary.is_root]
    # A root the graph lacked took every surviving bucket's summarized_in edge
    # with it (a document deletion drops the root), so its rebuild links every
    # bucket to it again, not only the ones whose text changed.
    root_was_missing = str(create_root_summary_id(data.dataset_id)) not in (
        data.existing_summary_texts
    )
    await _persist_bucket_edges(
        changed_buckets,
        data.dataset_id,
        unified_engine,
        link_documents=link_documents,
        relink=[
            summary
            for summary in desired.values()
            if not summary.is_root and summary not in changed_buckets
        ]
        if root_was_missing
        else [],
        ctx=ctx,
    )
    return changed


async def _persist_bucket_edges(
    buckets: list[GlobalContextSummary],
    dataset_id: str,
    unified_engine: Any,
    *,
    link_documents: bool,
    relink: list[GlobalContextSummary] = (),
    ctx: PipelineContext | None,
) -> None:
    """Link the buckets whose text was just written, and re-link ``relink`` to the root.

    ``relink`` holds unchanged buckets whose root edge is gone because the
    root was rebuilt; they get that edge only, their document edge was never
    touched. Every relationship goes out in one write, so the run costs one
    edge statement and one provenance-marker read however many kinds of edge
    it writes.

    The root id derives from the dataset id alone, so it never changes and an
    unchanged bucket's edges stay correct. The cost is that a run that crashes
    between the node write and here leaves those buckets unlinked until their
    text changes again; they are reached by vector search, not by traversal.
    """
    if not buckets and not relink:
        return

    root_id = str(create_root_summary_id(dataset_id))
    edges_by_relationship = {
        SUMMARIZED_IN: [(str(bucket.id), root_id) for bucket in (*buckets, *relink)]
    }
    if link_documents:
        edges_by_relationship[SUMMARIZES] = [
            (str(bucket.id), bucket.document_id) for bucket in buckets
        ]
    await persist_context_index_edges(edges_by_relationship, unified_engine, ctx=ctx)


def build_context_summaries(content: DatasetContent) -> list[GlobalContextSummary]:
    """The full desired index: one bucket per document that has entities, then the root.

    A dataset always gets its root, even when no document has an entity to name.
    The root is a vocabulary, not a ranking, so it is still worth having.
    """
    if not content.document_by_chunk:
        return []
    entities_by_document = _rank_document_entities(content)

    buckets = [
        GlobalContextSummary(
            id=create_document_bucket_id(content.dataset_id, document_id),
            text=_render_bucket_text(document_id, entities_by_document[document_id], content),
            dataset_id=content.dataset_id,
            level=0,
            is_root=False,
            document_id=document_id,
        )
        for document_id in sorted(
            entities_by_document,
            key=lambda document_id: (content.document_name(document_id), document_id),
        )
    ]
    return buckets + [
        GlobalContextSummary(
            id=create_root_summary_id(content.dataset_id),
            text=_render_root_text(content),
            dataset_id=content.dataset_id,
            level=1,
            is_root=True,
        )
    ]


def _rank_document_entities(content: DatasetContent) -> dict[str, _RankedEntities]:
    """Each document's top entities by tf-idf, in rank order. Documents with none are absent.

    IDF is over the dataset's chunks. Term frequency is counted as an integer
    before the single multiplication, so a score does not depend on the order
    chunk ids come out of a set. An entity in every chunk scores 0: it ranks last,
    but is still named while the bucket has room.
    """
    idf = compute_idf_from_counts(
        len(content.document_by_chunk),
        {entity_id: len(entity.chunk_ids) for entity_id, entity in content.entities.items()},
    )
    term_frequencies: dict[str, dict[str, int]] = {}
    for entity_id, entity in content.entities.items():
        for chunk_id in entity.chunk_ids:
            counts = term_frequencies.setdefault(content.document_by_chunk[chunk_id], {})
            counts[entity_id] = counts.get(entity_id, 0) + 1

    return {
        document_id: sorted(
            ((entity_id, count * idf[entity_id]) for entity_id, count in counts.items()),
            key=lambda scored: (-scored[1], content.entities[scored[0]].name, scored[0]),
        )[:DOCUMENT_BUCKET_MAX_ENTITIES]
        for document_id, counts in term_frequencies.items()
    }


def _rank_document_relations(
    ranked_entities: _RankedEntities,
    content: DatasetContent,
) -> list[str]:
    """Relation texts between the bucket's own entities, best first.

    ``min`` rather than ``sum``, so a relation ranks high only when both ends
    matter. Both endpoints are the bucket's own selected entities, so a relation
    line never names an entity the ``Type:`` lines above it leave out. The edge
    itself is any graph edge between those two entities, not necessarily one this
    document stated (or, in a shared database, this dataset): the graph read
    carries no edge provenance.

    The bucket's entity pairs are looked up, not each entity's relations walked:
    an entity in every chunk is named in almost every bucket, so walking would
    cost its relation count once per document.
    """
    score_by_entity = dict(ranked_entities)
    candidates = sorted(
        (-min(source_score, target_score), edge_text)
        for source_id, source_score in score_by_entity.items()
        for target_id, target_score in score_by_entity.items()
        for edge_text in content.relations.get((source_id, target_id), ())
    )
    return list(dict.fromkeys(edge_text for _, edge_text in candidates))[
        :DOCUMENT_BUCKET_MAX_RELATIONS
    ]


def _render_bucket_text(
    document_id: str,
    ranked_entities: _RankedEntities,
    content: DatasetContent,
) -> str:
    """Document name, then one ``Type: name, name`` line per type, then relations."""
    names_by_type: dict[str, list[str]] = {}
    for entity_id, _score in ranked_entities:  # already in rank order
        entity = content.entities[entity_id]
        names_by_type.setdefault(entity.type_name, []).append(entity.name)
    # dict keeps insertion order, so types come out ordered by their best entity.
    # No cap: the entity cap already bounds how many type lines there can be.
    type_lines = [f"{type_name}: {', '.join(names)}" for type_name, names in names_by_type.items()]
    relation_lines = _rank_document_relations(ranked_entities, content)
    return "\n".join((content.document_name(document_id), *type_lines, *relation_lines))


def _render_root_text(content: DatasetContent) -> str:
    """The dataset's vocabulary, not a prose summary: one name-dense line per label.

    A label with nothing to say is left out rather than embedded as an empty one.
    """
    labelled = (
        ("Topics", _topics_line(content)),
        ("Entities", _entities_line(content)),
        ("Documents", _documents_line(content)),
    )
    return "\n".join(f"{label}: {value}" for label, value in labelled if value)


def _topics_line(content: DatasetContent) -> str:
    """Entity types, by how many entities carry them."""
    type_counts = Counter(entity.type_name for entity in content.entities.values())
    topics = sorted(type_counts.items(), key=lambda counted: (-counted[1], counted[0]))
    return ", ".join(f"{name} ({count})" for name, count in topics[:ROOT_MAX_TYPES])


def _entities_line(content: DatasetContent) -> str:
    """The dataset's most widespread entity names, by how many documents mention them."""
    document_frequencies = {
        entity_id: len({content.document_by_chunk[chunk_id] for chunk_id in entity.chunk_ids})
        for entity_id, entity in content.entities.items()
    }
    entity_ids = sorted(
        document_frequencies,
        key=lambda entity_id: (
            -document_frequencies[entity_id],
            content.entities[entity_id].name,
            entity_id,
        ),
    )[:ROOT_MAX_ENTITIES]
    return ", ".join(content.entities[entity_id].name for entity_id in entity_ids)


def _documents_line(content: DatasetContent) -> str:
    """Document names, by how many chunks they have."""
    chunk_counts = Counter(content.document_by_chunk.values())
    documents = sorted(
        chunk_counts,
        key=lambda document_id: (
            -chunk_counts[document_id],
            content.document_name(document_id),
            document_id,
        ),
    )
    line = ", ".join(
        content.document_name(document_id) for document_id in documents[:ROOT_MAX_DOCUMENTS]
    )
    if len(documents) > ROOT_MAX_DOCUMENTS:
        # Never hide the dataset's size: the root is also read as a scope signal.
        line += f" (+{len(documents) - ROOT_MAX_DOCUMENTS} more)"
    return line
