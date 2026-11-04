"""Explicit deletion of one session: everything cognee keeps for it, owner only.

A session lives in two places. The session cache holds its content -- Q&A
turns, agent-trace steps, the session-context list (which also carries the
improve() watermarks) -- plus the Q&A vectors next to it; the relational DB
holds its dashboard row (``session_records``) and per-model usage rows
(``session_model_usage``). ``delete_session`` removes both, so the session
disappears from recall, from improve(), and from the sessions API alike.

Not touched, by design: lessons already distilled from the session into a
dataset (ordinary documents, removed with ``forget(data_id=...)``), and
``pipeline_runs`` rows that name the session (run history).

The Q&A vectors live where the session's turns were answered: with backend
access control on, each search turn indexes its Q&A in the isolated vector
store of the dataset it searched, and a recall over several datasets answers
once per dataset. ``remember()`` session writes index outside any dataset,
in the global store. Nothing records every dataset a session answered in
(the session record keeps only the first), so the delete strips the
session's scope tag in the global store and in every dataset the user can
read, plus the recorded dataset; the tag names this user and session, so a
store holding none of its rows is untouched. With access control off there
is only the global store.

The delete holds the session's improve claim, then its turn lock: an
improve() bridging the session finishes first, so its watermark writes
cannot recreate session context after the delete, and an answer being
written lands before the delete, never after it.

``delete_sessions`` deletes many sessions in one pass (forget(everything=True)):
each owner's datasets are resolved once, each vector store gets one
tag-removal call for all of its sessions, each relational table one delete
per owner, and every claim is taken up front.
"""

from collections.abc import Iterable
from contextlib import AsyncExitStack
from uuid import UUID

from sqlalchemy import delete, select

from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks import (
    acquire_improve_lock_many,
    release_improve_lock_many,
    session_turn_lock,
)
from cognee.infrastructure.locks.session_lock import improve_lock_keys
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_embeddings import delete_session_qa_vectors
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.shared.logging_utils import get_logger

from .models import SessionModelUsage, SessionRecord

logger = get_logger("delete_session")

# Session ids per relational statement, under every backend's bind-parameter cap.
_SESSIONS_PER_QUERY = 500

Session = tuple[UUID, str]


def _by_owner(sessions: list[Session]) -> dict[UUID, list[str]]:
    owners: dict[UUID, list[str]] = {}
    for user_id, session_id in sessions:
        owners.setdefault(user_id, []).append(session_id)
    return owners


def _chunks(session_ids: list[str]) -> Iterable[list[str]]:
    for start in range(0, len(session_ids), _SESSIONS_PER_QUERY):
        yield session_ids[start : start + _SESSIONS_PER_QUERY]


async def _delete_session_rows(sessions: list[Session]) -> set[Session]:
    """Delete the sessions' dashboard and usage rows; return the sessions that had any."""
    had_rows: set[Session] = set()
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        for user_id, session_ids in _by_owner(sessions).items():
            for chunk in _chunks(session_ids):
                for model in (SessionModelUsage, SessionRecord):
                    scope = (model.user_id == user_id, model.session_id.in_(chunk))
                    found = await db_session.execute(select(model.session_id).where(*scope))
                    had_rows.update((user_id, session_id) for session_id in found.scalars())
                    await db_session.execute(delete(model).where(*scope))
        await db_session.commit()
    return had_rows


async def _existing_datasets(dataset_ids: set[UUID]) -> set[UUID]:
    """The ids that still name a dataset; a deleted one invalidated its sessions already."""
    if not dataset_ids:
        return set()
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        found = await db_session.execute(select(Dataset.id).where(Dataset.id.in_(dataset_ids)))
        return set(found.scalars())


async def _recorded_datasets(sessions: list[Session]) -> dict[Session, UUID]:
    """The dataset recorded on each session's row, for the sessions that have one."""
    recorded: dict[Session, UUID] = {}
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        for user_id, session_ids in _by_owner(sessions).items():
            for chunk in _chunks(session_ids):
                rows = await db_session.execute(
                    select(SessionRecord.session_id, SessionRecord.dataset_id).where(
                        SessionRecord.user_id == user_id,
                        SessionRecord.session_id.in_(chunk),
                        SessionRecord.dataset_id.is_not(None),
                    )
                )
                recorded.update(((user_id, row.session_id), row.dataset_id) for row in rows)
    return recorded


async def _session_vector_datasets(sessions: list[Session]) -> dict[UUID, list[Session]]:
    """The datasets whose vector stores may hold the sessions' Q&A vectors, with their sessions.

    For each session: every dataset its owner can read, which covers every
    dataset a recall of theirs searched, plus the dataset recorded on the
    session row, in case read access has since been revoked. A dataset id
    parsed from the client's session id is never trusted: it names a dataset
    only if it is readable. Each owner's datasets are resolved once.
    """
    targets: dict[UUID, list[Session]] = {}
    for user_id, session_ids in _by_owner(sessions).items():
        readable = await get_all_user_permission_datasets(await get_user(user_id), "read")
        for dataset in readable:
            targets.setdefault(dataset.id, []).extend(
                (user_id, session_id) for session_id in session_ids
            )
    recorded = await _recorded_datasets(sessions)
    existing = await _existing_datasets(set(recorded.values()))
    for session, dataset_id in recorded.items():
        if dataset_id in existing and session not in targets.get(dataset_id, []):
            targets.setdefault(dataset_id, []).append(session)
    return targets


async def delete_session(user_id: UUID, session_id: str) -> bool:
    """Delete one of ``user_id``'s sessions; True if anything existed, False if nothing did.

    Raises when session memory is off (``CACHING=false``): there is no cache
    to delete from, and answering False would read as "no such session".
    Cache and vector-store errors propagate; a retry deletes what is left.
    """
    return await delete_sessions([(user_id, session_id)]) > 0


async def delete_sessions(sessions: Iterable[Session]) -> int:
    """Delete each (owner id, session id) pair whole; return how many existed.

    Same contract as ``delete_session``, for many sessions at once: it raises
    with ``CACHING=false``, and cache, vector-store and relational errors
    propagate, so a retry deletes what is left. The caller decides whose
    sessions these are; this deletes exactly the pairs it is given.
    """
    sessions = list(dict.fromkeys(sessions))
    if any(not session_id for _, session_id in sessions):
        raise CogneeValidationError(
            message="delete_session needs an explicit session_id.",
            name="SessionPreconditionError",
        )
    session_manager = get_session_manager()
    if not session_manager.is_available:
        raise CogneeValidationError(
            message=(
                "Session memory is disabled (CACHING=false), so there is no session to delete. "
                "Enable CACHING to use sessions."
            ),
            name="SessionPreconditionError",
        )
    if not sessions:
        return 0

    dataset_targets = (
        await _session_vector_datasets(sessions) if backend_access_control_enabled() else {}
    )
    claim_keys = [
        key
        for user_id, session_ids in _by_owner(sessions).items()
        for key in improve_lock_keys(session_ids, user_id)
    ]
    await acquire_improve_lock_many(claim_keys)
    try:
        async with AsyncExitStack() as turn_locks:
            # One fixed order, so two batch deletes sharing sessions cannot deadlock.
            for user_id, session_id in sorted(sessions, key=lambda s: (str(s[0]), s[1])):
                await turn_locks.enter_async_context(session_turn_lock(str(user_id), session_id))
            # The cache and the global store first: a dataset context's vector
            # config outlives its block, so afterwards the global store is out
            # of reach.
            cache_existed = await session_manager.delete_sessions(
                [(str(user_id), session_id) for user_id, session_id in sessions]
            )
            for dataset_id, dataset_sessions in dataset_targets.items():
                # The store resolves under the dataset's owner whoever enters
                # (no permission check is asked for), so any session owner will do.
                async with set_database_global_context_variables(
                    dataset_id, dataset_sessions[0][0]
                ):
                    await delete_session_qa_vectors(
                        (str(user_id), session_id) for user_id, session_id in dataset_sessions
                    )
            had_rows = await _delete_session_rows(sessions)
    finally:
        await release_improve_lock_many(claim_keys)

    existed = sum(
        1 for session, in_cache in zip(sessions, cache_existed) if in_cache or session in had_rows
    )
    logger.info(
        "Deleted %d session(s), %d of which existed.",
        len(sessions),
        existed,
    )
    return existed
