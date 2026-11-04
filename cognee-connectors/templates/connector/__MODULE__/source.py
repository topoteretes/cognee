"""The __DISPLAY_NAME__ DLT source: the records cognee ingests, rendered as documents.

A skeleton on the pattern every connector in this repository follows: a ``dlt``
resource that yields one row per document, opted into cognee's document ingestion
path through the attributes :mod:`cognee_connectors_core.dlt_conventions` names.
Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
``primary_key="id"``::

    import cognee
    from __MODULE__ import __PROVIDER___source

    await cognee.remember(
        __PROVIDER___source(token="..."),
        dataset_name="my___PROVIDER__",
        primary_key="id",
        write_disposition="merge",
    )
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any

from cognee_connectors_core import dlt_conventions

logger = logging.getLogger(__name__)


def __PROVIDER___source(
    token: str | None = None,
    resource_name: str = "__PROVIDER___documents",
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource yielding one row per __DISPLAY_NAME__ document.

    Args:
        token: The credential the provider's API takes (an OAuth access token or an
            API key).
        resource_name: Stable dlt resource name, also the staging table and the key
            of the incremental state. Every source syncing into one dataset needs
            its own name: two sources sharing a name share state.
        check_active: Optional host authorization checkpoint during extraction.
    """
    import dlt

    if not token:
        raise ValueError("__PROVIDER___source requires a token")

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def __PROVIDER___documents() -> Iterator[dict[str, Any]]:
        yield from dlt_conventions.guarded_rows(_iter_rows(token), check_active)

    resource = __PROVIDER___documents()
    # Opt into the document ingestion path: each row becomes a text document that
    # flows through normal cognify. cognee reads these markers by attribute name.
    setattr(resource, dlt_conventions.DOCUMENT_SOURCE_ATTR, "__PROVIDER__")
    setattr(resource, dlt_conventions.PIPELINE_SCOPE_ATTR, resource_name)
    return resource


def _iter_rows(token: str) -> Iterator[dict[str, Any]]:
    """Fetch the provider's records and render each as a document row.

    A row carries ``id`` (stable across runs), ``title``, ``content`` (the text cognee
    extracts from) and ``url``; optionally ``dlt_conventions.NODE_SET_COLUMN`` (a list
    of node-set names, namespaced under the source tag) and
    ``dlt_conventions.STRUCTURE_COLUMN`` (where the record sits in its tree). Emit
    ``{"id": ..., "_deleted": True}`` for a record that disappeared since the last run.
    """
    raise NotImplementedError("fetch __DISPLAY_NAME__ records and yield one row per document")
