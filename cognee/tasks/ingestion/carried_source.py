"""The typed handoff between the incremental pipeline wrapper and ``ingest_data``.

The wrapper saves each item to storage and hashes its bytes to resolve the
item's dedup identity before the task chain runs. Without a handoff,
``ingest_data`` pays the same upload and the same read-back per item — on the
S3 backend a duplicate PUT plus a duplicate HEAD+GET per file. The wrapper
publishes what it did as a :class:`CarriedSource` on ``ctx.extras``;
``ingest_data`` recovers it through :func:`find_carried_source`.

Matching is two-phase because item identity is not always stable: the in-chain
``resolve_data_directories`` re-creates string paths, so a path item reaches
``ingest_data`` as a different object. Uploads and other non-string items match
by ``id()`` before any storage work; path items (whose save is a pass-through
with no I/O) match by the stored path the save returns.
"""

from dataclasses import dataclass
from typing import Optional

from cognee.modules.ingestion import StoredFile

# ``ctx.extras`` key the wrapper publishes under. ``ctx`` is copied per item by
# ``run_tasks``, and the wrapper runs once per item, so the entry can only
# describe that item.
CARRIED_SOURCE_KEY = "ingest_carried_source"


@dataclass
class CarriedSource:
    """Storage work already done for one data item."""

    # id() of the exact object the wrapper inspected — the first matching key.
    data_item_id: int
    # Where the payload landed, plus the metadata computed from its bytes
    # (None for items whose bytes never passed through this process).
    stored: StoredFile


def publish_carried_source(ctx, data_item, stored: StoredFile) -> None:
    if ctx is None:
        return
    ctx.extras[CARRIED_SOURCE_KEY] = CarriedSource(data_item_id=id(data_item), stored=stored)


def find_carried_source(ctx, *, data_item=None, file_path: str = None) -> Optional[StoredFile]:
    """The wrapper's :class:`StoredFile` for this item, or None.

    Pass ``data_item`` to match by object identity (before any storage work),
    or ``file_path`` to match by where a pass-through save resolved the item
    (after it). A miss means the caller does the storage work itself — the
    handoff is an optimization, never a correctness dependency.
    """
    extras = getattr(ctx, "extras", None) if ctx is not None else None
    if not extras:
        return None

    carried = extras.get(CARRIED_SOURCE_KEY)
    if not isinstance(carried, CarriedSource):
        return None

    if data_item is not None and carried.data_item_id == id(data_item):
        return carried.stored
    if file_path is not None and carried.stored.file_path == file_path:
        return carried.stored

    return None
