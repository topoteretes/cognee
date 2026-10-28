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
"""

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

from .metrics import get_session_row
from .models import SessionModelUsage, SessionRecord

logger = get_logger("delete_session")


async def _delete_session_rows(user_id: UUID, session_id: str) -> int:
    """Delete the session's dashboard and usage rows; return how many rows went."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        deleted = 0
        for model in (SessionModelUsage, SessionRecord):
            result = await db_session.execute(
                delete(model).where(model.user_id == user_id, model.session_id == session_id)
            )
            deleted += result.rowcount or 0
        await db_session.commit()
    return deleted


async def _dataset_exists(dataset_id: UUID) -> bool:
    """Whether the dataset still exists; its own forget() invalidated its sessions already."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        found = (
            await db_session.execute(select(Dataset.id).where(Dataset.id == dataset_id))
        ).scalar_one_or_none()
    return found is not None


async def _session_vector_datasets(user_id: UUID, session_id: str) -> list[UUID]:
    """The datasets whose vector stores may hold the session's Q&A vectors.

    Every dataset the user can read, which covers every dataset a recall of
    theirs searched, plus the dataset recorded on the session row, in case
    read access has since been revoked. A dataset id parsed from the client's
    session id is never trusted: it names a dataset only if it is readable.
    """
    dataset_ids = [
        dataset.id
        for dataset in await get_all_user_permission_datasets(await get_user(user_id), "read")
    ]
    row = await get_session_row(session_id=session_id, user_id=user_id)
    if row is not None and row.dataset_id is not None and await _dataset_exists(row.dataset_id):
        dataset_ids.append(row.dataset_id)
    return list(dict.fromkeys(dataset_ids))


async def delete_session(user_id: UUID, session_id: str) -> bool:
    """Delete one of ``user_id``'s sessions; True if anything existed, False if nothing did.

    Raises when session memory is off (``CACHING=false``): there is no cache
    to delete from, and answering False would read as "no such session".
    Cache and vector-store errors propagate; a retry deletes what is left.
    """
    if not session_id:
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

    dataset_ids = (
        await _session_vector_datasets(user_id, session_id)
        if backend_access_control_enabled()
        else []
    )
    claim_keys = improve_lock_keys([session_id], user_id)
    await acquire_improve_lock_many(claim_keys)
    try:
        async with session_turn_lock(str(user_id), session_id):
            # The cache and the global store first: a dataset context's vector
            # config outlives its block, so afterwards the global store is out
            # of reach.
            cache_deleted = await session_manager.delete_session(
                user_id=str(user_id), session_id=session_id
            )
            for dataset_id in dataset_ids:
                async with set_database_global_context_variables(dataset_id, user_id):
                    await delete_session_qa_vectors(user_id=str(user_id), session_id=session_id)
            rows_deleted = await _delete_session_rows(user_id, session_id)
    finally:
        await release_improve_lock_many(claim_keys)

    logger.info(
        "Deleted session %s of user %s: cache content %s, %d relational row(s).",
        session_id,
        user_id,
        "removed" if cache_deleted else "absent",
        rows_deleted,
    )
    return cache_deleted or rows_deleted > 0
