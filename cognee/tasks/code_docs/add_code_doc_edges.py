"""Link docstring chunks to the code graph and sweep the ones a re-run no longer yields.

The code docs chain (``code_docs_tail`` on the code graph routes, the memify
pipeline in ``cognee/memify_pipelines/cognify_code_docs.py``) stores a
repository's docstrings -- the docs Enola wrote to ``docs.jsonl`` -- as plain
``DocumentChunk`` nodes, one per documented symbol or module, text headed
``file:line``, under one ``CodeDocument`` per file (per package directory for
a module doc). Plain chunks on purpose: every retriever that searches
``DocumentChunk_text`` already finds them, so no retrieval change is needed.
What a plain chunk lacks is the hop into the code graph, and this task adds it:
a ``describes`` edge from each chunk to the ``CodeSymbol`` (for a module doc,
the ``CodeModule``) its ``external_metadata`` names. Edges are
written by id through the graph engine, the way ``add_code_graph_edges`` writes
typed relations; no code graph node is ever emitted here, so the code graph's
nodes, hashes and edges stay exactly as its own pipeline left them.

Ids are deterministic (``code_document_id`` for a document, the content hash of
the text for a chunk), so a re-run over unchanged sources yields the same ids:
the edge diff adds nothing and the sweep finds nothing. A changed docstring
yields a chunk with a new id; its predecessor, every document the run no
longer produced, and the documents of a repository the run no longer covers
(its CodeRepository forgotten or renamed, or its docstrings gone) are swept
here, from the graph and from the vector collections they were embedded into.
What hangs off a stale chunk goes with it: its TextSummary, and in entity
mode the Entity and EntityType nodes it alone owned -- through the shared
last-owner provenance planner when the chunk carries source refs (stored under
a pipeline context with a data item, as on the code graph routes), else by
reading the ownership from the graph.

Only the extraction's own batch (CodeDocsBatch) is swept against: it records
which repositories were read, from which snapshot. A repository the extraction
did not read -- a lost clone, a gone staging directory, a snapshot another
enola run wrote over -- keeps its stored docs, and so does one whose graph has
moved to another snapshot since the extraction read it. A plain list from any
other caller is linked and never swept against.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import data_item_id, parse_source_ref_key
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.external_metadata import decode_external_metadata
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.methods.delete_chunks_incremental import delete_chunks_incremental
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_docs.extract_code_docs import node_properties
from cognee.tasks.code_docs.models import DESCRIBES_RELATIONSHIP, CodeDocument
from cognee.tasks.code_graph.extract_code_graph import (
    _SWEEP_CHUNK_SIZE,
    _invalidate_code_graph_snapshot,
)
from cognee.tasks.summarization.models import TextSummary

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")

# Stored type names (DataPoint.type is the class name).
_CODE_DOCUMENT_TYPE = CodeDocument.__name__
_CHUNK_TYPE = DocumentChunk.__name__
_SUMMARY_TYPE = TextSummary.__name__
_ENTITY_TYPE = Entity.__name__
_ENTITY_TYPE_TYPE = EntityType.__name__
# The one graph read covers these stored types. A filtered read returns an
# edge only when both of its endpoints match, so the code node types are in
# for the describes edges and the summaries for their made_from edges. The
# read is the dataset's every DocumentChunk, text included, not just the code
# docs: there is no property filter below the type on every backend. Walking
# instead from the repositories' CodeDocuments (get_neighborhood, two hops)
# measured 2-4x faster on Ladybug for a 20k-chunk corpus next to a 300-chunk
# repository, and 4-5x slower for a 5k-chunk repository on its own, so the
# single read stays until a dataset shows the corpus cost.
_READ_TYPES = [
    _CODE_DOCUMENT_TYPE,
    _CHUNK_TYPE,
    _SUMMARY_TYPE,
    "CodeSymbol",
    "CodeModule",
    # One node per repository: whether it still exists, and which snapshot its
    # graph holds now (the sweep keeps a repository whose graph moved on).
    "CodeRepository",
]
# The second read entity mode's sweep needs (chunk -contains-> Entity -is_a->
# EntityType), paid only when a stale chunk without source refs has an entity.
_ENTITY_READ_TYPES = [_CHUNK_TYPE, _ENTITY_TYPE, _ENTITY_TYPE_TYPE]
# Up to this many stale chunks are asked for their neighbours one by one (a
# point lookup each) before the entity read is paid; above it, the read is
# the cheaper question.
_NEIGHBOR_LOOKUP_LIMIT = 64


@dataclass
class SweepCounts:
    """Stale nodes one sweep removed, by kind."""

    documents: int = 0
    chunks: int = 0
    # TextSummaries made only from stale chunks, swept by hand.
    summaries: int = 0
    # Entity and EntityType nodes only stale chunks held, swept by hand.
    entities: int = 0
    # Nodes the provenance planner retired with a chunk that owned them
    # (its summary, its entities): the chunks themselves are under ``chunks``.
    owned: int = 0

    @property
    def total(self) -> int:
        return self.documents + self.chunks + self.summaries + self.entities + self.owned


@dataclass
class LinkCounts:
    """What one link-and-sweep pass did: describes edges written, kept, not written; the sweep."""

    edges: int = 0
    present: int = 0
    # Chunks whose code node the graph does not hold.
    unresolved: int = 0
    # Chunks the graph does not hold themselves (the batch was not stored).
    unstored: int = 0
    swept: SweepCounts = field(default_factory=SweepCounts)


async def link_code_docs(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    sweep: bool = True,
    repo: str | None = None,
) -> LinkCounts:
    """Write the batch's ``describes`` edges, sweep the stale docstring nodes, return the counts.

    The work behind add_code_doc_edges (which passes the batch through) and
    code_docs_tail (which reports these counts). See add_code_doc_edges for the
    batch and the sweep's scope.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    counts = LinkCounts()
    chunks, documents = _collect_batch(data_points)
    read_snapshots = getattr(data_points, "read_snapshots", None)
    if not chunks and not documents and (data_points or read_snapshots is None):
        # Nothing of ours to link, and no extraction record to sweep against:
        # a pipeline whose batch holds no code docs, or an empty plain list.
        # No graph read either. An empty CodeDocsBatch is a code docs run that
        # derived nothing, and still sweeps below.
        return counts

    try:
        graph_engine = await get_graph_engine()
        raw_nodes, raw_edges = await graph_engine.get_filtered_graph_data([{"type": _READ_TYPES}])
        nodes = {str(node_id): node_properties(properties) for node_id, properties in raw_nodes}
        edges = [(str(item[0]), str(item[1]), str(item[2])) for item in raw_edges if len(item) >= 3]
        existing_edge_keys = set(edges)

        new_edges = []
        for chunk_id, (chunk, code_doc) in chunks.items():
            if chunk_id not in nodes:
                counts.unstored += 1
                continue
            target_id = _edge_target(code_doc, nodes)
            if target_id is None:
                counts.unresolved += 1
            elif (chunk_id, target_id, DESCRIBES_RELATIONSHIP) in existing_edge_keys:
                counts.present += 1
            else:
                new_edges.append(_describes_edge(chunk_id, target_id, chunk))
        counts.edges = len(new_edges)
        logger.info(
            "Code doc edge delta: %d new, %d already present.", counts.edges, counts.present
        )
        if counts.unresolved:
            logger.warning(
                "%d docstring chunk(s) name a code node the graph does not hold; not linked.",
                counts.unresolved,
            )
        if counts.unstored:
            logger.warning(
                "%d docstring chunk(s) of the batch are not in the graph (store_code_docs runs "
                "before this task); not linked.",
                counts.unstored,
            )
        if new_edges:
            await graph_engine.add_edges(new_edges)
            await _register_edges(new_edges, ctx)

        if sweep and read_snapshots is not None:
            counts.swept = await _sweep_stale_code_docs(
                graph_engine,
                _sweep_scope(repo, nodes, read_snapshots, _batch_ids_by_repo(chunks, documents)),
                set(documents),
                set(chunks),
                nodes,
                edges,
            )
        elif sweep:
            logger.debug("Code docs linked, not swept: the batch carries no extraction record.")
    finally:
        # Edges on code nodes changed, or may have on a partial failure: the
        # code retriever's parsed snapshot must not outlive them.
        _invalidate_code_graph_snapshot(ctx)
    return counts


async def add_code_doc_edges(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    sweep: bool = True,
    repo: str | None = None,
) -> list[DataPoint]:
    """Write the batch's ``describes`` edges, then sweep the stale docstring nodes.

    Runs after store_code_docs and receives the whole run: the documents and
    chunks in the embed-only chain, or TextSummary objects (chunk under
    ``made_from``, document under the chunk's ``is_part_of``) when
    cognify_code_doc_chunks ran first. Only edges the graph does not hold yet
    are written, the same new-vs-existing diff add_code_graph_edges does, so a
    re-run adds nothing.

    ``sweep`` deletes the stored code docs the run did not produce, within the
    run's scope: the repository ``repo`` names, or, without a filter, every
    repository that has CodeDocuments -- named in the batch or not, so a
    repository whose CodeRepository node was forgotten or renamed, or whose
    docstrings all vanished, loses its documents too. In scope go the
    CodeDocuments the batch lacks with their chunks, the chunks of kept
    documents whose ids the batch lacks (their docstring changed), the
    TextSummaries made from deleted chunks only, and the Entity/EntityType
    nodes only deleted chunks held, from the graph and from their vector
    collections; other repositories and non-code documents are never
    considered. An empty batch still sweeps.

    The sweep runs only against the extraction's own batch: a CodeDocsBatch
    records the repositories extract_code_docs read and the snapshot each was
    read from. Left out of the scope, with their stored docs kept, are a
    repository that still has its CodeRepository node but was not read (its
    snapshot docs were unavailable), one whose graph holds another snapshot by
    now (another run loaded it meanwhile), and one whose docs from this batch
    are not all in the graph (the batch was not stored first). A plain list
    from another caller is linked and never swept against.

    The batch is the whole run: every task of the chain is a coroutine, and
    the pipeline hands a coroutine's result to the next task in one call, the
    same object (a generator's output would arrive in ``batch_size`` pieces,
    and a sweep per piece would delete the run's other pieces). ``repo`` must
    therefore match the extraction's. Passthrough: returns data_points
    unchanged.
    """
    await link_code_docs(data_points, ctx=ctx, sweep=sweep, repo=repo)
    return data_points


def _code_doc_metadata(chunk: DocumentChunk) -> dict[str, Any] | None:
    """The ``code_doc`` block of a chunk's external_metadata, None on any other chunk."""
    metadata = decode_external_metadata(chunk.external_metadata)
    code_doc = metadata.get("code_doc") if metadata else None
    return code_doc if isinstance(code_doc, dict) else None


def _collect_batch(
    data_points: list[DataPoint],
) -> tuple[dict[str, tuple[DocumentChunk, dict[str, Any]]], dict[str, CodeDocument]]:
    """The batch's docstring chunks (with their metadata) and CodeDocuments, by string id.

    Chunks arrive directly, or under a TextSummary's ``made_from``; a document
    arrives directly, or as a chunk's ``is_part_of``.
    """
    chunks: dict[str, tuple[DocumentChunk, dict[str, Any]]] = {}
    documents: dict[str, CodeDocument] = {}
    for point in data_points:
        if isinstance(point, CodeDocument):
            documents[str(point.id)] = point
            continue
        chunk = point if isinstance(point, DocumentChunk) else getattr(point, "made_from", None)
        if not isinstance(chunk, DocumentChunk):
            continue
        code_doc = _code_doc_metadata(chunk)
        if code_doc is None:
            continue
        chunks[str(chunk.id)] = (chunk, code_doc)
        if isinstance(chunk.is_part_of, CodeDocument):
            documents[str(chunk.is_part_of.id)] = chunk.is_part_of
    return chunks, documents


def _edge_target(code_doc: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str | None:
    """Id of the code node a chunk describes, when the graph holds it.

    A symbol doc names its CodeSymbol (``symbol_id``), a module doc its
    CodeModule (``module_id``): the node extract_code_docs joined the
    docs.jsonl line to by its enola id.
    """
    for key in ("symbol_id", "module_id"):
        node_id = code_doc.get(key)
        if node_id is not None:
            node_id = str(node_id)
            return node_id if node_id in nodes else None
    return None


def _describes_edge(
    chunk_id: str, target_id: str, chunk: DocumentChunk
) -> tuple[str, str, str, dict[str, Any]]:
    """One edge tuple in the shape build_code_graph_edges writes, plus the chunk's index."""
    return (
        chunk_id,
        target_id,
        DESCRIBES_RELATIONSHIP,
        {
            "chunk_index": chunk.chunk_index,
            "source_node_id": chunk_id,
            "target_node_id": target_id,
            "relationship_name": DESCRIBES_RELATIONSHIP,
            "edge_text": DESCRIBES_RELATIONSHIP.replace("_", " "),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


async def _register_edges(edges: list[tuple], ctx: Optional["PipelineContext"]) -> None:
    """Record new edges in the relational rollback ledger, as add_code_graph_edges does.

    Only a run over a persisted data item has a ledger row to key them on. A
    memify run has none (its item is the ``[{}]`` seed), so this is a no-op
    there and pipeline rollback relies on graph provenance instead. The
    graph engine takes the endpoints as strings; the ledger's columns are
    UUIDs, as the code graph's own edge tuples carry them.
    """
    data_id = data_item_id(getattr(ctx, "data_item", None))
    if (
        ctx is None
        or getattr(ctx, "user", None) is None
        or getattr(ctx, "dataset", None) is None
        or data_id is None
        or getattr(ctx, "pipeline_run_id", None) is None
    ):
        return
    from cognee.modules.graph.methods import upsert_edges

    await upsert_edges(
        [
            (UUID(str(source)), UUID(str(target)), name, props)
            for source, target, name, props in edges
        ],
        tenant_id=ctx.user.tenant_id,
        user_id=ctx.user.id,
        dataset_id=ctx.dataset.id,
        data_id=data_id,
        pipeline_run_id=ctx.pipeline_run_id,
    )


def _batch_ids_by_repo(
    chunks: Mapping[str, tuple[DocumentChunk, dict[str, Any]]],
    documents: Mapping[str, CodeDocument],
) -> dict[str, set[str]]:
    """The batch's document and chunk ids, by the repository they belong to."""
    ids: dict[str, set[str]] = {}
    for document_id, document in documents.items():
        ids.setdefault(document.repo, set()).add(document_id)
    for chunk_id, (_chunk, code_doc) in chunks.items():
        if isinstance(code_doc.get("repo"), str):
            ids.setdefault(code_doc["repo"], set()).add(chunk_id)
    return ids


def _sweep_scope(
    repo: str | None,
    nodes: dict[str, dict[str, Any]],
    read_snapshots: Mapping[str, str | None],
    batch_ids: Mapping[str, set[str]],
) -> set[str]:
    """The repositories whose stored docs the batch accounts for.

    With a ``repo`` filter the run covered that repository alone. Without one
    it covered every CodeRepository in the dataset, so every repository that
    has CodeDocuments is in scope -- one named in the batch, one whose
    CodeRepository node is gone (forgotten or renamed), one whose docstrings
    all vanished: a repository absent from the whole-run batch has no current
    docs. The scope is read off the CodeDocuments, not the CodeRepository
    nodes, which is why a forgotten repository is still found and swept.

    Out of scope again, with its stored docs kept, is a repository that still
    has its CodeRepository node and:

    - was not read by the extraction (``read_snapshots``, its own record:
      the snapshot docs were unavailable), whatever its snapshot directory
      looks like by now;
    - holds another snapshot by now than the one its docs were read from
      (another run loaded it in the meantime and stored that snapshot's
      docs, which this batch would sweep);
    - has docs in this batch that the graph read does not hold (the batch
      was not stored before it was linked), which this batch would replace
      with nothing.
    """
    if repo is not None:
        scope = {repo}
    else:
        scope = {
            properties["repo"]
            for properties in nodes.values()
            if properties.get("type") == _CODE_DOCUMENT_TYPE
            and isinstance(properties.get("repo"), str)
        }
    repositories = {
        properties["name"]: properties
        for properties in nodes.values()
        if properties.get("type") == "CodeRepository" and isinstance(properties.get("name"), str)
    }
    not_read, moved, unstored = set(), set(), set()
    for name in scope & repositories.keys():
        if name not in read_snapshots:
            not_read.add(name)
        elif repositories[name].get("last_snapshot_id") != read_snapshots[name]:
            moved.add(name)
        elif any(node_id not in nodes for node_id in batch_ids.get(name, ())):
            unstored.add(name)
    for names, why in (
        (not_read, "the extraction did not read the docs of the snapshot their code graph holds"),
        (moved, "their code graph moved to another snapshot after the extraction read it"),
        (unstored, "the run's docs of them are not in the graph (they were not stored first)"),
    ):
        if names:
            logger.warning(
                "Code docs of %s kept as stored: %s.",
                ", ".join(repr(name) for name in sorted(names)),
                why,
            )
    return scope - not_read - moved - unstored


async def _sweep_stale_code_docs(
    graph_engine,
    repos: set[str],
    batch_document_ids: set[str],
    batch_chunk_ids: set[str],
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, str, str]],
) -> SweepCounts:
    """Delete the docstring nodes of these repositories that the batch did not produce.

    ``nodes``/``edges`` are the pre-write graph read. Stale documents are the
    stored CodeDocuments of a repository in scope whose ids the batch lacks;
    stale chunks are theirs (reverse ``is_part_of``) plus the chunks of kept
    documents whose ids the batch lacks. A stale chunk that carries its own
    chunk-scoped source ref is retired through delete_chunks_incremental, the
    shared last-owner planner, which removes what only it owned (summary,
    entities, edges) from graph and vectors; the rest is swept by hand: the
    chunks, the TextSummaries made only from stale chunks, the Entity and
    EntityType nodes only stale chunks held (_stale_entities). Deleting a node
    deletes its edges, so the describes edges go with the chunks.
    """
    repo_documents = {
        node_id
        for node_id, properties in nodes.items()
        if properties.get("type") == _CODE_DOCUMENT_TYPE and properties.get("repo") in repos
    }
    stale_documents = repo_documents - batch_document_ids

    chunk_document: dict[str, str] = {}
    summary_chunks: dict[str, set[str]] = {}
    for source, target, relationship in edges:
        source_type = nodes.get(source, {}).get("type")
        if relationship == "is_part_of" and source_type == _CHUNK_TYPE:
            if target in repo_documents:
                chunk_document[source] = target
        elif relationship == "made_from" and source_type == _SUMMARY_TYPE:
            summary_chunks.setdefault(source, set()).add(target)
    stale_chunks = {
        chunk_id
        for chunk_id, document_id in chunk_document.items()
        if document_id in stale_documents or chunk_id not in batch_chunk_ids
    }
    counts = SweepCounts(documents=len(stale_documents), chunks=len(stale_chunks))
    if not stale_documents and not stale_chunks:
        return counts

    owned = await _owned_chunks(graph_engine, stale_chunks)
    owned_chunks = {chunk_id for chunk_ids in owned.values() for chunk_id in chunk_ids}
    for (dataset_id, data_id), chunk_ids in sorted(owned.items()):
        result = await delete_chunks_incremental(sorted(chunk_ids), dataset_id, data_id)
        deleted = {str(node_id) for node_id in getattr(result, "deleted_node_ids", None) or ()}
        counts.owned += len(deleted - owned_chunks)
    unowned_chunks = stale_chunks - owned_chunks

    stale_summaries = {
        summary_id
        for summary_id, chunk_ids in summary_chunks.items()
        if chunk_ids <= stale_chunks and not chunk_ids & owned_chunks
    }
    stale_entities, stale_types = await _stale_entities(graph_engine, unowned_chunks, stale_chunks)
    counts.summaries = len(stale_summaries)
    counts.entities = len(stale_entities) + len(stale_types)

    # Vectors first, then the graph nodes in the order the next sweep finds
    # them back: what hangs off a chunk (summaries through made_from, entity
    # types and entities through is_a and contains) before the chunks, the
    # chunks before the documents they are found through. A run that fails
    # part of the way leaves stale nodes the next sweep finds again, never a
    # vector without a node that only retrieval still sees.
    await _delete_vectors(
        {
            DocumentChunk.vector_collection("text"): sorted(unowned_chunks),
            TextSummary.vector_collection("text"): sorted(stale_summaries),
            Entity.vector_collection("name"): sorted(stale_entities),
            EntityType.vector_collection("name"): sorted(stale_types),
            CodeDocument.vector_collection("name"): sorted(stale_documents),
        }
    )
    stale = (
        sorted(stale_summaries)
        + sorted(stale_types)
        + sorted(stale_entities)
        + sorted(unowned_chunks)
        + sorted(stale_documents)
    )
    for start in range(0, len(stale), _SWEEP_CHUNK_SIZE):
        await graph_engine.delete_nodes(stale[start : start + _SWEEP_CHUNK_SIZE])
    logger.info(
        "Swept %d stale code document(s), %d chunk(s) (%d through the provenance planner, "
        "with %d node(s) they owned), %d summary(ies) and %d entity node(s).",
        counts.documents,
        counts.chunks,
        len(owned_chunks),
        counts.owned,
        counts.summaries,
        counts.entities,
    )
    return counts


async def _owned_chunks(graph_engine, chunk_ids: set[str]) -> dict[tuple[str, str], set[str]]:
    """The stale chunks stamped with their own chunk-scoped source ref, by (dataset id, data id).

    A chunk add_data_points stored under a pipeline context with a persisted
    data item (the code graph routes) carries
    ``source_ref:v2:<dataset>:<data>:<chunk>``, and so does everything its
    extraction produced -- its summary, its entities, their edges -- which is
    exactly what the planner retires with it. A chunk stored without a data
    item (a memify run) carries no ref and is swept by hand; so is every chunk
    on a backend that stores no provenance.
    """
    if not chunk_ids:
        return {}
    try:
        delete_data = await graph_engine.get_node_delete_data(sorted(chunk_ids))
    except UnsupportedProvenanceCapability:
        return {}
    owned: dict[tuple[str, str], set[str]] = {}
    for chunk_id, data in (delete_data or {}).items():
        for key in getattr(data, "source_ref_keys", None) or ():
            try:
                parsed = parse_source_ref_key(key)
            except ValueError:
                continue
            if parsed.version == 2 and str(parsed.chunk_id) == str(chunk_id):
                owned.setdefault((str(parsed.dataset_id), str(parsed.data_id)), set()).add(
                    str(chunk_id)
                )
                break
    return owned


async def _has_entity_neighbors(graph_engine, chunk_ids: set[str]) -> bool:
    """Whether any of these chunks ``contains`` an Entity or EntityType (a point lookup each)."""
    for chunk_id in sorted(chunk_ids):
        for row in await graph_engine.get_connections(chunk_id) or ():
            if len(row) < 3:
                continue
            if node_properties(row[1]).get("relationship_name") != "contains":
                continue
            for node in (row[0], row[2]):
                if node_properties(node).get("type") in (_ENTITY_TYPE, _ENTITY_TYPE_TYPE):
                    return True
    return False


async def _stale_entities(
    graph_engine, unowned_chunks: set[str], stale_chunks: set[str]
) -> tuple[set[str], set[str]]:
    """The Entity and EntityType nodes only stale chunks hold: (entity ids, type ids).

    Entity mode stores, under each chunk, the Entity nodes it ``contains`` and
    the EntityType each ``is_a``; graph merging stores equal nodes once, so a
    chunk's entity may be another chunk's too. A chunk without source refs
    cannot hand this to the planner, so the ownership is read from the graph:
    an Entity goes when every chunk that ``contains`` it is stale, an
    EntityType when every Entity ``is_a`` it goes and no surviving chunk
    ``contains`` it directly. Shared entities stay. Relationship edges between
    two surviving entities that only a stale chunk stated cannot be attributed
    without refs and stay.

    The read of the dataset's chunks, entities and types is paid only when a
    stale chunk has an entity: a few stale chunks are asked one by one first
    (_has_entity_neighbors); above _NEIGHBOR_LOOKUP_LIMIT the read is the
    cheaper question. An embed-only dataset never pays it.
    """
    if not unowned_chunks:
        return set(), set()
    if len(unowned_chunks) <= _NEIGHBOR_LOOKUP_LIMIT and not await _has_entity_neighbors(
        graph_engine, unowned_chunks
    ):
        return set(), set()

    raw_nodes, raw_edges = await graph_engine.get_filtered_graph_data(
        [{"type": _ENTITY_READ_TYPES}]
    )
    types = {
        str(node_id): node_properties(properties).get("type") for node_id, properties in raw_nodes
    }
    # entity or type -> the chunks that contain it; type -> the entities that are it.
    contained_in: dict[str, set[str]] = {}
    instances: dict[str, set[str]] = {}
    for item in raw_edges:
        if len(item) < 3:
            continue
        source, target, relationship = str(item[0]), str(item[1]), str(item[2])
        if relationship == "contains" and types.get(source) == _CHUNK_TYPE:
            if types.get(target) in (_ENTITY_TYPE, _ENTITY_TYPE_TYPE):
                contained_in.setdefault(target, set()).add(source)
        elif (
            relationship == "is_a"
            and types.get(source) == _ENTITY_TYPE
            and types.get(target) == _ENTITY_TYPE_TYPE
        ):
            instances.setdefault(target, set()).add(source)

    candidates = {node_id for node_id, chunks in contained_in.items() if chunks & unowned_chunks}
    stale_entities = {
        node_id
        for node_id in candidates
        if types.get(node_id) == _ENTITY_TYPE and contained_in[node_id] <= stale_chunks
    }
    type_candidates = {node_id for node_id in candidates if types.get(node_id) == _ENTITY_TYPE_TYPE}
    type_candidates |= {
        type_id for type_id, entities in instances.items() if entities & stale_entities
    }
    stale_types = {
        type_id
        for type_id in type_candidates
        if contained_in.get(type_id, set()) <= stale_chunks
        and instances.get(type_id, set()) <= stale_entities
    }
    return stale_entities, stale_types


async def _delete_vectors(stale_by_collection: dict[str, list[str]]) -> None:
    """Drop the swept nodes' embeddings from the collections add_data_points indexed them into.

    Deleting the graph node alone would leave its vector behind, and the chunk
    retrievers (CHUNKS, RAG_COMPLETION) read ``DocumentChunk_text`` payloads
    without consulting the graph, so a replaced docstring would keep being
    returned next to its replacement. The sweep deletes the vectors before the
    graph nodes: if it fails in between, a node left without its vector is
    found and swept again by the next run through the graph, while a vector
    left without its node would be served forever. A collection that was never
    written (a graph_only run) is skipped.
    """
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    stale_by_collection = {name: ids for name, ids in stale_by_collection.items() if ids}
    if not stale_by_collection:
        return
    vector_engine = await get_vector_engine_async()
    for collection, ids in stale_by_collection.items():
        if not await vector_engine.has_collection(collection):
            continue
        for start in range(0, len(ids), _SWEEP_CHUNK_SIZE):
            await vector_engine.delete_data_points(
                collection, ids[start : start + _SWEEP_CHUNK_SIZE]
            )
