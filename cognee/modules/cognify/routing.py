"""Cognify routing policy: which task route a data item takes.

One pure, synchronous function — no I/O, no configuration — importable by
everything that needs the decision: cognify() (to resolve per-item task
lists), the dry-run estimator (so cost estimates can never drift from
execution), and, via document_class_for, classification itself.

The routing fact is written onto each Data record at add time
(external_metadata / extension); this module only reads it.
"""

from enum import Enum

from cognee.tasks.documents.classify_documents import document_class_for
from cognee.modules.data.processing.document_types import DltRowDocument, DltSourceDocument


class CognifyRoute(Enum):
    """The task routes cognify can send a data item down."""

    STANDARD = "standard"
    DLT_SOURCE = "dlt_source"
    DLT_ROW_LEGACY = "dlt_row_legacy"


_ROUTE_BY_DOCUMENT_CLASS = {
    DltSourceDocument: CognifyRoute.DLT_SOURCE,
    DltRowDocument: CognifyRoute.DLT_ROW_LEGACY,
}


def cognify_route_for(data_item) -> CognifyRoute:
    """The cognify route for one data item. Pure function of the record."""
    return _ROUTE_BY_DOCUMENT_CLASS.get(document_class_for(data_item), CognifyRoute.STANDARD)
