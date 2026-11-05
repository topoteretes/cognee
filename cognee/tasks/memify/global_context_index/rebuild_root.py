"""Write the ``from_entities`` root again after a document deletion.

A document deletion drops the dataset's root (``remove_document_index``) because
its text names the deleted document. Until the next ``improve()`` the dataset
would then have bucket rows but no root row in the routing collection, so a
broad question has nothing to land on and the dataset can lose the routing to
another one. The ``from_entities`` root is a rendered vocabulary: one graph read
and one embedding of a short text, no LLM, so it is written again at once.

Only that method's root. The ``llm`` method's root is an LLM summary over bucket
summaries that still describe the deleted document; it stays dropped until its
own build. The two are told apart by shape: a ``from_entities`` index has one
bucket per document whose id derives from the dataset and document
(``create_document_bucket_id``), an ``llm`` index has clustering ids.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.search.dataset_routing import sync_dataset_routes
from cognee.shared.logging_utils import get_logger
from cognee.tasks.summarization.models import GlobalContextSummary

from .build_document_index import _persist_bucket_edges, build_root_summary
from .ids import create_document_bucket_id, create_root_summary_id
from .load_dataset_content import load_dataset_content
from .models import DatasetContent
from .persist import (
    delete_context_index_nodes,
    persist_context_summaries,
    safe_delete_context_vectors,
)

logger = get_logger("global_context_index")


async def rebuild_document_index_root(dataset_id: UUID | str, user) -> GlobalContextSummary | None:
    """Write the dataset's ``from_entities`` root from the documents it has now; never raises.

    For a dataset that had an index: ``remove_document_index`` reports that,
    and the caller runs this only then, since a never-indexed dataset must stay
    without index nodes (routing searches it always). Runs after a document's
    memory is gone, in the dataset's database context, like the deletion before
    it. Returns the root written, or ``None`` when there was nothing to write:
    the index is the ``llm`` method's, or no document with chunks survives (the
    last document went). The forget that calls it is done by then, so a
    failure here, an embedding provider that cannot be reached first of all, is
    a warning and leaves the root absent until the next ``improve()``, as before:
    a root node the failure left behind is taken out again, since the build
    compares text and would take it for complete.
    """
    try:
        return await _rebuild_root(str(dataset_id), user)
    except Exception:
        logger.warning(
            "Global context index: the root of dataset %s could not be rebuilt after the "
            "deletion; broad queries route to the dataset again after the next improve.",
            dataset_id,
            exc_info=True,
        )
        return None


async def _rebuild_root(dataset_id: str, user) -> GlobalContextSummary | None:
    ctx = PipelineContext(user=user, dataset=SimpleNamespace(id=UUID(dataset_id)))
    content = await load_dataset_content(ctx=ctx)
    if not index_is_from_entities(content):
        return None
    root = build_root_summary(content)
    if root is None:
        return None
    buckets = stored_document_buckets(content)

    unified_engine = await get_unified_engine()
    root_id = str(root.id)
    # The deletion dropped the root node and its vector. Should a root still be
    # stored, its vector goes first, as in the build: the stores keep a
    # conflicting row's vector, and the id never changes.
    if root_id in content.existing_summary_texts:
        await safe_delete_context_vectors(unified_engine.vector, [root_id])
    try:
        await persist_context_summaries([root], ctx)
        # The root took every bucket's summarized_in edge with it.
        await _persist_bucket_edges(
            [], dataset_id, unified_engine, link_documents=False, relink=buckets, ctx=ctx
        )
        # The buckets go in as stored, so the mirror sees them unchanged and
        # copies the root's vector only. Their texts shift a little with the
        # deletion (tf-idf over fewer documents); the next build writes those.
        await sync_dataset_routes(dataset_id, [*buckets, root], unified_engine.vector)
    except Exception:
        # The node is written before its text is embedded, and the edges after
        # both, so a failure can leave a root with no vector or no bucket
        # edges. The build compares text, and this root's text is the one it
        # computes, so it would leave such a root as it is: unembedded, the
        # mirror cannot copy its vector and every later build fails. Take the
        # root out again, and the next build writes it as one that was missing.
        await delete_context_index_nodes(unified_engine, [root_id])
        raise
    logger.info("Global context index: root of dataset %s rebuilt after a deletion", dataset_id)
    return root


def index_is_from_entities(content: DatasetContent) -> bool:
    """Whether every stored index node is one the dataset's documents derive.

    The ``from_entities`` method writes the root and one bucket per document,
    all by derived id, so an index node no surviving document accounts for is
    the ``llm`` method's clustering bucket. An emptied index (the deletion took
    the last bucket) passes: the documents that remain still get their root.
    """
    derivable = {str(create_root_summary_id(content.dataset_id))} | {
        str(create_document_bucket_id(content.dataset_id, document_id))
        for document_id in set(content.document_by_chunk.values())
    }
    return all(node_id in derivable for node_id in content.existing_summary_texts)


def stored_document_buckets(content: DatasetContent) -> list[GlobalContextSummary]:
    """The dataset's document buckets as the index stores them now.

    One per surviving document whose derived bucket id is an index node. A
    document whose entities were all extracted from the deleted document's
    chunks, or that never had any, has no bucket and is named by the root alone.
    """
    buckets = []
    for document_id in sorted(set(content.document_by_chunk.values())):
        bucket_id = create_document_bucket_id(content.dataset_id, document_id)
        text = content.existing_summary_texts.get(str(bucket_id))
        if text:
            buckets.append(
                GlobalContextSummary(
                    id=bucket_id,
                    text=text,
                    dataset_id=content.dataset_id,
                    level=0,
                    is_root=False,
                    document_id=document_id,
                )
            )
    return buckets
