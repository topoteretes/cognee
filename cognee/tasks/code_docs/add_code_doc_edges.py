"""Link docstring chunks to the code graph and sweep the ones a re-run no longer yields.

The code docs memify pipeline (``cognee/memify_pipelines/cognify_code_docs.py``)
stores a repository's docstrings as plain ``DocumentChunk`` nodes -- one per
documented symbol, text headed ``file:line`` -- under one ``CodeDocument`` per
source file. Plain chunks on purpose: every retriever that searches
``DocumentChunk_text`` already finds them, so no retrieval change is needed.
What a plain chunk lacks is the hop into the code graph, and this task adds it:
a ``describes`` edge from each chunk to the ``CodeSymbol`` (for a module
docstring, the ``CodeModule``) its ``external_metadata`` names. Edges are
written by id through the graph engine, the way ``add_code_graph_edges`` writes
typed relations; no code graph node is ever emitted here, so the code graph's
nodes, hashes and edges stay exactly as its own pipeline left them.

Ids are deterministic (``code_document_id`` for a document, the content hash of
the text for a chunk), so a re-run over unchanged sources yields the same ids:
the edge diff adds nothing and the sweep finds nothing. A changed docstring
yields a chunk with a new id; its predecessor, and every document the run no
longer produced, is swept here, from the graph and from the vector collections
it was embedded into.
"""

import posixpath
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from cognee.infrastructure.databases.provenance import data_item_id
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.external_metadata import decode_external_metadata
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_docs.extract_code_docs import node_properties
from cognee.tasks.code_docs.models import DESCRIBES_RELATIONSHIP, CodeDocument
from cognee.tasks.code_graph.extract_code_graph import (
    _SWEEP_CHUNK_SIZE,
    _invalidate_code_graph_snapshot,
    fact_node_id,
)
from cognee.tasks.summarization.models import TextSummary

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")

# Stored type names (DataPoint.type is the class name).
_CODE_DOCUMENT_TYPE = CodeDocument.__name__
_CHUNK_TYPE = DocumentChunk.__name__
_SUMMARY_TYPE = TextSummary.__name__
# The one graph read covers these stored types. A filtered read returns an
# edge only when both of its endpoints match, so the code node types are in
# for the describes edges and the summaries for their made_from edges. The
# read is the dataset's every DocumentChunk, text included, not just the code
# docs: there is no property filter below the type on every backend. Walking
# instead from the repositories' CodeDocuments (get_neighborhood, two hops)
# measured 2-4x faster on Ladybug for a 20k-chunk corpus next to a 300-chunk
# repository, and 4-5x slower for a 5k-chunk repository on its own, so the
# single read stays until a dataset shows the corpus cost.
_READ_TYPES = [_CODE_DOCUMENT_TYPE, _CHUNK_TYPE, _SUMMARY_TYPE, "CodeSymbol", "CodeModule"]


async def add_code_doc_edges(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    sweep: bool = True,
) -> list[DataPoint]:
    """Write the batch's ``describes`` edges, then sweep the stale docstring nodes.

    Runs after add_data_points and receives its output: the documents and
    chunks in the embed-only chain, or TextSummary objects (chunk under
    ``made_from``, document under the chunk's ``is_part_of``) when
    extract_graph_and_summarize ran first. Only edges the graph does not hold
    yet are written, the same new-vs-existing diff add_code_graph_edges does,
    so a re-run adds nothing. ``sweep`` deletes, per repository seen in the
    batch, the CodeDocuments the batch did not produce with their chunks, the
    chunks of kept documents whose ids the batch lacks (their docstring
    changed), and the TextSummaries made from deleted chunks only, from the
    graph and from their vector collections; other repositories and non-code
    documents are never considered. The batch is the whole run: every task of
    the chain is a coroutine, and the pipeline hands a coroutine's result to
    the next task in one call (a generator's output would arrive in
    ``batch_size`` pieces, and a sweep per piece would delete the run's other
    pieces). Passthrough: returns data_points unchanged.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if not data_points:
        # extract_code_docs found no code graph or no docstrings: nothing to
        # link, and no repository to sweep.
        return data_points

    chunks, documents = _collect_batch(data_points)
    if not chunks and not documents:
        # Composed into a pipeline whose batch holds no code docs: nothing of
        # ours to touch, so no graph read either.
        return data_points
    repos = {code_doc.get("repo") for _chunk, code_doc in chunks.values()}
    repos |= {document.repo for document in documents.values()}
    repos.discard(None)

    try:
        graph_engine = await get_graph_engine()
        raw_nodes, raw_edges = await graph_engine.get_filtered_graph_data([{"type": _READ_TYPES}])
        nodes = {str(node_id): node_properties(properties) for node_id, properties in raw_nodes}
        edges = [(str(item[0]), str(item[1]), str(item[2])) for item in raw_edges if len(item) >= 3]
        existing_edge_keys = set(edges)

        new_edges = []
        present = unresolved = 0
        for chunk_id, (chunk, code_doc) in chunks.items():
            target_id = _edge_target(code_doc, nodes)
            if target_id is None:
                unresolved += 1
            elif (chunk_id, target_id, DESCRIBES_RELATIONSHIP) in existing_edge_keys:
                present += 1
            else:
                new_edges.append(_describes_edge(chunk_id, target_id, chunk, code_doc))
        logger.info("Code doc edge delta: %d new, %d already present.", len(new_edges), present)
        if unresolved:
            logger.warning(
                "%d docstring chunk(s) name a code node the graph does not hold; not linked.",
                unresolved,
            )
        if new_edges:
            await graph_engine.add_edges(new_edges)
            await _register_edges(new_edges, ctx)

        if sweep:
            await _sweep_stale_code_docs(
                graph_engine, repos, set(documents), set(chunks), nodes, edges
            )
    finally:
        # Edges on code nodes changed, or may have on a partial failure: the
        # code retriever's parsed snapshot must not outlive them.
        _invalidate_code_graph_snapshot(ctx)
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

    A symbol docstring names its CodeSymbol (``symbol_id``). A module docstring
    names the CodeModule of the file's directory when extract_code_docs found
    one (``module_id``); failing that, the module is derived the way the code
    graph names it -- enola module facts are directory paths, the repo root is
    "." -- and used only when that node was read.
    """
    for key in ("symbol_id", "module_id"):
        node_id = code_doc.get(key)
        if node_id is not None:
            node_id = str(node_id)
            return node_id if node_id in nodes else None
    repo, file_path = code_doc.get("repo"), code_doc.get("file")
    if code_doc.get("symbol_kind") != "module" or not isinstance(repo, str):
        return None
    if not isinstance(file_path, str):
        return None
    module_id = str(fact_node_id(repo, "module", posixpath.dirname(file_path) or "."))
    return module_id if nodes.get(module_id, {}).get("type") == "CodeModule" else None


def _describes_edge(
    chunk_id: str, target_id: str, chunk: DocumentChunk, code_doc: dict[str, Any]
) -> tuple[str, str, str, dict[str, Any]]:
    """One edge tuple in the shape build_code_graph_edges writes, plus the doc provenance."""
    return (
        chunk_id,
        target_id,
        DESCRIBES_RELATIONSHIP,
        {
            "doc_source": code_doc.get("doc_source"),
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
    there and pipeline rollback relies on graph provenance instead.
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
        edges,
        tenant_id=ctx.user.tenant_id,
        user_id=ctx.user.id,
        dataset_id=ctx.dataset.id,
        data_id=data_id,
        pipeline_run_id=ctx.pipeline_run_id,
    )


async def _sweep_stale_code_docs(
    graph_engine,
    repos: set[str],
    batch_document_ids: set[str],
    batch_chunk_ids: set[str],
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, str, str]],
) -> tuple[int, int, int]:
    """Delete the docstring nodes of these repositories that the batch did not produce.

    ``nodes``/``edges`` are the pre-write graph read. Stale documents are the
    stored CodeDocuments of a batch repository whose ids the batch lacks; stale
    chunks are theirs (reverse ``is_part_of``) plus the chunks of kept
    documents whose ids the batch lacks; a TextSummary goes when every chunk
    it was ``made_from`` goes. Deleting a node deletes its edges, so the
    describes edges go with the chunks. Returns (documents, chunks, summaries)
    removed.
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
    stale_summaries = {
        summary_id for summary_id, chunk_ids in summary_chunks.items() if chunk_ids <= stale_chunks
    }

    stale = sorted(stale_documents) + sorted(stale_chunks) + sorted(stale_summaries)
    for start in range(0, len(stale), _SWEEP_CHUNK_SIZE):
        await graph_engine.delete_nodes(stale[start : start + _SWEEP_CHUNK_SIZE])
    await _delete_vectors(
        {
            CodeDocument.vector_collection("name"): sorted(stale_documents),
            DocumentChunk.vector_collection("text"): sorted(stale_chunks),
            TextSummary.vector_collection("text"): sorted(stale_summaries),
        }
    )
    if stale:
        logger.info(
            "Swept %d stale code document(s), %d chunk(s) and %d summary(ies).",
            len(stale_documents),
            len(stale_chunks),
            len(stale_summaries),
        )
    return len(stale_documents), len(stale_chunks), len(stale_summaries)


async def _delete_vectors(stale_by_collection: dict[str, list[str]]) -> None:
    """Drop the swept nodes' embeddings from the collections add_data_points indexed them into.

    Deleting the graph node alone would leave its vector behind, and the chunk
    retrievers (CHUNKS, RAG_COMPLETION) read ``DocumentChunk_text`` payloads
    without consulting the graph, so a replaced docstring would keep being
    returned next to its replacement. Graph first, then vectors, the order
    cognee's own deletion uses (delete_from_graph_and_vector): a node left
    without its vector is still found through the graph, a vector left without
    its node is served forever. A collection that was never written (a
    graph_only run) is skipped.
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
