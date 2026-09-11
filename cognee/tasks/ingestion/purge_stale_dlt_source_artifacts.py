"""Purge a DLT source's previously derived artifacts before re-emission.

Runs right after classify_documents in the DLT cognify route. A manifest's
Data id is stable (dlt_source:{dataset}:{source}), so reprocessing a changed
source re-emits its rows under the same identity — without this purge, chunks
of deleted rows and old versions of changed rows would linger in the graph
and vector stores and keep surfacing in search. Deleting by (dataset, data
id) is idempotent and a no-op on first processing.

The window where the source is absent shrinks to inside this cognify run
(between this task and add_data_points/extract_dlt_source_edges completing);
between add and cognify, search returns the OLD rows — stale beats absent.
A crash after the purge leaves the source absent until the retry, which
re-runs the purge harmlessly and rebuilds.
"""

from typing import TYPE_CHECKING, List, Optional

from cognee.modules.data.processing.document_types import Document, DltSourceDocument
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("purge_stale_dlt_source_artifacts")


async def purge_stale_dlt_source_artifacts(
    documents: List[Document],
    ctx: Optional["PipelineContext"] = None,
) -> List[Document]:
    manifest_docs = [doc for doc in documents if isinstance(doc, DltSourceDocument)]
    if not manifest_docs:
        return documents

    if ctx is None or ctx.dataset is None or ctx.user is None:
        raise ValueError(
            "purge_stale_dlt_source_artifacts requires a PipelineContext with "
            "dataset and user — it deletes per (dataset, data id) and must "
            "never guess the scope."
        )

    from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
        delete_data_nodes_and_edges,
    )

    for doc in manifest_docs:
        # Document.id IS the manifest's Data id (classify_documents builds it
        # so). Deletion re-authorizes with "delete" permission; a denial
        # propagates loudly — silently skipping would present stale rows as
        # fresh, which is worse than a failed run. Reprocessing a changed DLT
        # source therefore requires delete permission on the dataset.
        deleted_elements = await delete_data_nodes_and_edges(ctx.dataset.id, doc.id, ctx.user.id)
        # Session answers that used the purged elements would replay the stale
        # content after the re-emission (the "revise" contamination of COG-5835);
        # invalidate them best-effort.
        try:
            from cognee.modules.session_lifecycle.invalidate_sessions import (
                invalidate_sessions_for_deleted_data,
            )

            await invalidate_sessions_for_deleted_data(
                ctx.dataset.id,
                deleted_elements.node_ids,
                deleted_elements.edge_ids,
                user_id=ctx.user.id,
            )
        except Exception as error:
            logger.warning("Session invalidation after DLT purge failed (non-fatal): %s", error)
        logger.info(
            "Purged prior derived artifacts of DLT source data item %s before re-emission.",
            doc.id,
        )

    return documents
