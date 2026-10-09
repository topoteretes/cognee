"""Authorize and atomically bind typed session writes to one dataset."""

from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session
from cognee.modules.session_lifecycle.models import SessionRecord


async def bind_typed_session_dataset(user, session_id: str, dataset_id: UUID):
    dataset = await get_authorized_dataset(user, UUID(str(dataset_id)), "write")
    if dataset is None:
        raise ValueError("Selected session dataset is unavailable")
    # The upsert only fills an unset binding, so concurrent first writers to
    # different datasets cannot reassign a session. Verify before cache writes.
    await ensure_and_touch_session(session_id=session_id, user_id=user.id, dataset_id=dataset.id)
    async with get_relational_engine().get_async_session() as session:
        bound, status = (
            await session.execute(
                select(SessionRecord.dataset_id, SessionRecord.status).where(
                    SessionRecord.session_id == session_id, SessionRecord.user_id == user.id
                )
            )
        ).one()
    if bound != dataset.id:
        raise ValueError("Session belongs to another dataset; use a new session_id")
    if status != "running":
        raise ValueError("Session is already ended; use a new session_id")
    return dataset
