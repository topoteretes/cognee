"""Public API surface for dataset versioning (event sourcing + checkpoints)."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from cognee.modules.versioning.models.Checkpoint import Checkpoint
from cognee.modules.versioning.operations.create_checkpoint import create_checkpoint
from cognee.modules.versioning.operations.get_events import get_event_log
from cognee.modules.versioning.operations.log_event import log_version_event
from cognee.modules.versioning.operations.time_travel import get_nodes_at_time
from cognee.modules.versioning.operations.undo_forget import UndoForgetResult, undo_forget


async def get_version_history(
    dataset_id: UUID,
    *,
    data_id: Optional[UUID] = None,
    operation: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """Return the event log for a dataset, newest first.

    Args:
        dataset_id: The dataset to inspect.
        data_id: Narrow to a specific data item (optional).
        operation: Filter to ``"ADD"``, ``"COGNIFY"``, or ``"FORGET"`` (optional).
        limit: Max events to return (default 100).

    Returns:
        List of event dicts, each with ``id``, ``operation``, ``dataset_id``,
        ``data_id``, ``created_at``, ``undone_at``, ``node_slugs``, ``edge_slugs``.

    Example::

        import cognee
        history = await cognee.get_version_history(dataset_id)
        for event in history:
            print(event["operation"], event["created_at"])
    """
    return await get_event_log(
        dataset_id, data_id=data_id, operation=operation, limit=limit
    )


async def snapshot(
    dataset_id: UUID,
    *,
    user_id: Optional[UUID] = None,
    label: Optional[str] = None,
) -> Checkpoint:
    """Create a lightweight checkpoint of the alive node/edge set.

    Checkpoints store node slugs and edge relationship names as JSON arrays.
    They are cheap to create and support time-travel diffing.

    Args:
        dataset_id: The dataset to snapshot.
        user_id: Owner of the checkpoint (optional).
        label: Human-readable tag, e.g. ``"after-ingestion-v2"`` (optional).

    Returns:
        The newly created :class:`Checkpoint` instance.

    Example::

        import cognee
        cp = await cognee.snapshot(dataset_id, label="before-reindex")
        print(cp.id, cp.created_at)
    """
    return await create_checkpoint(dataset_id, user_id=user_id, label=label)


async def undo_forget_data(
    dataset_id: UUID,
    *,
    data_id: Optional[UUID] = None,
    event_id: Optional[UUID] = None,
) -> UndoForgetResult:
    """Reverse the most recent FORGET event for a dataset.

    This operation clears the ``deleted_at`` timestamp on
    ``GraphRelationshipLedger`` rows captured in the FORGET event payload,
    restoring the ledger audit trail.

    .. note::
        Graph and vector data is hard-deleted and cannot be automatically
        restored.  After calling this function, re-run ``cognee.add()`` and
        ``cognee.cognify()`` with the original data files to fully restore the
        knowledge graph.  The returned :class:`UndoForgetResult` includes the
        affected node and edge slugs for reference.

    Args:
        dataset_id: The dataset whose last FORGET should be reversed.
        data_id: Narrow to a specific data item (optional).
        event_id: Target a specific event ID (optional).

    Returns:
        :class:`UndoForgetResult` with ``node_slugs``, ``edge_slugs``, and
        ``ledger_rows_restored``.

    Raises:
        ValueError: If no matching unresolved FORGET event is found.

    Example::

        import cognee
        result = await cognee.undo_forget_data(dataset_id)
        print(result.ledger_rows_restored, "ledger rows restored")
        print("Re-ingest these node slugs:", result.node_slugs[:5])
    """
    return await undo_forget(dataset_id, data_id=data_id, event_id=event_id)


async def time_travel(
    dataset_id: UUID,
    target_time: datetime,
) -> Dict:
    """Return the alive node-slug set for *dataset_id* at *target_time*.

    Uses the nearest checkpoint before *target_time* as the base, then
    replays ADD and FORGET events up to *target_time* to reconstruct the
    exact node set that was alive at that moment.

    Args:
        dataset_id: The dataset to inspect.
        target_time: Point in time to reconstruct. Timezone-aware recommended;
            naive datetimes are treated as UTC.

    Returns:
        Dict with ``alive_node_slugs`` (sorted list), ``dataset_id``,
        ``target_time``, ``checkpoint_id``, and ``checkpoint_time``.

    Example::

        import cognee
        from datetime import datetime, timezone, timedelta

        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        state = await cognee.time_travel(dataset_id, one_hour_ago)
        print(state["alive_node_slugs"])
    """
    return await get_nodes_at_time(dataset_id, target_time)
