"""Session-memory invalidation on document/dataset deletion (COG-5947).

Deleting a document scrubs the graph and vector stores, but session memory —
cached Q&A turns, distilled session-context lessons, and their vector index —
kept quoting the deleted content, so completions asserted facts that no longer
exist (COG-5835). This module removes the contaminated session state:

* ``invalidate_sessions_for_dataset`` — dataset-level deletes drop every
  session attributed to the dataset (the ticket's baseline semantics).
* ``invalidate_sessions_for_deleted_data`` — single-document deletes remove
  only the turns whose recorded ``used_graph_element_ids`` intersect the
  elements the delete removed, then follow the intra-session provenance chain
  (turn -> feedback entry -> context lesson -> later turns that consumed the
  lesson) to a fixpoint.

Both are best-effort by contract: they must never fail the delete that
triggered them, so callers wrap them in non-fatal try/except and all cache
lookups fail open.

Known limits (documented, deliberate): agent-trace entries carry context text
without element ids and are not matched; the tapes cache backend is
append-only and never sees deletes; sessions predating dataset attribution are
only found through the ``{default_session_id}_{dataset_id}`` naming. Data-level
deletes also scan every dataset-unattributed session (``dataset_id IS NULL``);
element-id matching keeps that scan precise (COG-6292).
"""

from uuid import UUID

from cognee.infrastructure.locks import session_lock
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import (
    get_persisted_qa_count,
    save_persisted_qa_count,
)
from cognee.shared.logging_utils import get_logger

from .metrics import list_sessions_for_dataset, list_unattributed_sessions

logger = get_logger("invalidate_sessions")


async def invalidate_sessions_for_dataset(dataset_id: UUID) -> dict:
    """Delete every session attributed to the dataset from the session cache.

    Used by dataset-level deletes (``forget(dataset=...)``, ``empty_dataset``),
    where all session content derives from data that is now gone.
    """
    session_manager = get_session_manager()
    if not session_manager.is_available:
        return {"sessions_considered": 0, "sessions_deleted": 0}

    sessions = await list_sessions_for_dataset(dataset_id)
    deleted = 0
    for user_id, session_id in sessions:
        try:
            if await session_manager.delete_session(user_id=str(user_id), session_id=session_id):
                deleted += 1
        except Exception as error:
            logger.warning(
                "Session invalidation: failed to delete session %s for user %s (non-fatal): %s",
                session_id,
                user_id,
                error,
            )

    if sessions:
        logger.info(
            "Session invalidation: deleted %d/%d session(s) for dataset %s",
            deleted,
            len(sessions),
            dataset_id,
        )
    return {"sessions_considered": len(sessions), "sessions_deleted": deleted}


async def invalidate_sessions_for_deleted_data(
    dataset_id: UUID,
    deleted_node_ids: set[str],
    deleted_edge_ids: set[str],
    user_id: UUID | None = None,
) -> dict:
    """Remove session entries contaminated by a deleted data item.

    A turn is contaminated when its recorded ``used_graph_element_ids``
    intersect the node/edge identities the delete removed. Contamination then
    propagates within the session: feedback entries referencing a contaminated
    turn, context lessons distilled from contaminated feedback, and later
    turns that consumed a contaminated lesson (``used_session_context_ids``),
    iterated to a fixpoint.

    Dataset-attributed sessions are unioned with every dataset-unattributed
    session (``dataset_id IS NULL``). An unscoped search runs in the plain
    default session, which carries no dataset attribution. Element-id matching
    keeps the wider scan precise (COG-6292). ``user_id`` is accepted for
    callers and does not filter the unattributed query.
    """
    totals = {"sessions_considered": 0, "qa_entries_deleted": 0, "context_entries_deleted": 0}
    if not deleted_node_ids and not deleted_edge_ids:
        return totals

    session_manager = get_session_manager()
    if not session_manager.is_available:
        return totals

    sessions = list(await list_sessions_for_dataset(dataset_id))
    seen = {(str(session_user_id), session_id) for session_user_id, session_id in sessions}
    for candidate in await list_unattributed_sessions():
        if (str(candidate[0]), candidate[1]) not in seen:
            sessions.append(candidate)
    totals["sessions_considered"] = len(sessions)

    for user_id, session_id in sessions:
        try:
            counts = await _invalidate_session_entries(
                session_manager,
                user_id=str(user_id),
                session_id=session_id,
                deleted_node_ids=deleted_node_ids,
                deleted_edge_ids=deleted_edge_ids,
            )
            totals["qa_entries_deleted"] += counts[0]
            totals["context_entries_deleted"] += counts[1]
        except Exception as error:
            logger.warning(
                "Session invalidation: targeted cleanup failed for session %s, user %s "
                "(non-fatal): %s",
                session_id,
                user_id,
                error,
            )

    if totals["qa_entries_deleted"] or totals["context_entries_deleted"]:
        logger.info(
            "Session invalidation: removed %d QA and %d context entr(ies) referencing "
            "deleted data in dataset %s",
            totals["qa_entries_deleted"],
            totals["context_entries_deleted"],
            dataset_id,
        )
    return totals


async def _invalidate_session_entries(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    deleted_node_ids: set[str],
    deleted_edge_ids: set[str],
) -> tuple[int, int]:
    """Targeted cleanup of one session. Returns (qa_deleted, context_deleted)."""
    entries = await session_manager.get_session(user_id=user_id, session_id=session_id)
    if not entries:
        return (0, 0)

    contaminated_qa_ids: set[str] = set()
    for entry in entries:
        used = entry.used_graph_element_ids or {}
        used_nodes = set(used.get("node_ids") or [])
        used_edges = set(used.get("edge_ids") or [])
        if (used_nodes & deleted_node_ids) or (used_edges & deleted_edge_ids):
            if entry.qa_id:
                contaminated_qa_ids.add(entry.qa_id)

    if not contaminated_qa_ids:
        return (0, 0)

    context_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )

    # Propagate contamination along the intra-session provenance chain until
    # nothing new is found: turn -> feedback (referenced_qa_ids) -> lesson
    # (source_feedback_ids) -> later turns (used_session_context_ids).
    contaminated_feedback_ids: set[str] = set()
    contaminated_context_ids: set[str] = set()
    changed = True
    while changed:
        changed = False
        for context_entry in context_entries:
            entry_id = context_entry.get("id")
            if not entry_id:
                continue
            kind = context_entry.get("kind")
            if kind == "feedback" and entry_id not in contaminated_feedback_ids:
                if set(context_entry.get("referenced_qa_ids") or []) & contaminated_qa_ids:
                    contaminated_feedback_ids.add(entry_id)
                    changed = True
            elif kind == "context" and entry_id not in contaminated_context_ids:
                if set(context_entry.get("source_feedback_ids") or []) & contaminated_feedback_ids:
                    contaminated_context_ids.add(entry_id)
                    changed = True
        for entry in entries:
            if not entry.qa_id or entry.qa_id in contaminated_qa_ids:
                continue
            if set(entry.used_session_context_ids or []) & contaminated_context_ids:
                contaminated_qa_ids.add(entry.qa_id)
                changed = True

    qa_deleted = 0
    for qa_id in contaminated_qa_ids:
        if await session_manager.delete_qa(user_id=user_id, qa_id=qa_id, session_id=session_id):
            qa_deleted += 1

    context_deleted = 0
    for entry_id in contaminated_feedback_ids | contaminated_context_ids:
        if await session_manager.delete_session_context_entry(
            user_id=user_id, entry_id=entry_id, session_id=session_id
        ):
            context_deleted += 1

    # Clamp the persist watermark to the surviving entry count. A watermark
    # above the count reads as "session was rebuilt" and makes the next
    # improve() re-persist the whole session from the start (see the
    # session_persist_watermark module docstring). The clamp is a fresh
    # recount under the same (session_id, "update_qa") lock the QA
    # delete/update flows use — never the pre-delete snapshot — so
    # concurrent invalidations serialize and each writes the true count.
    # Writes only ever lower the watermark; a too-low value is safe (the
    # next improve() re-persists a little extra, deduped at add()).
    if qa_deleted:
        async with session_lock(session_id, "update_qa"):
            surviving = await session_manager.get_session(user_id=user_id, session_id=session_id)
            remaining = len(surviving) if surviving else 0
            watermark = await get_persisted_qa_count(session_manager, user_id, session_id)
            if watermark > remaining:
                await save_persisted_qa_count(session_manager, user_id, session_id, remaining)

    return (qa_deleted, context_deleted)
