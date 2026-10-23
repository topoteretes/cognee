"""``get_session_dataset_id`` reads attribution the way ``list_sessions_for_dataset`` writes it.

A session is attributed to a dataset by the ``dataset_id`` backfilled on its row,
else by the dataset id a per-dataset default session carries as its session_id
suffix. A session nothing attributed answers None, as does a session with no row.
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.session_lifecycle.metrics import (
    ensure_and_touch_session,
    get_session_dataset_id,
)
from cognee.modules.session_lifecycle.models import SessionRecord


@pytest_asyncio.fixture
async def session_table():
    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)
    created: list[tuple[str, object]] = []

    yield created

    async with engine.get_async_session() as session:
        for session_id, user_id in created:
            row = await session.get(SessionRecord, (session_id, user_id))
            if row:
                await session.delete(row)
        await session.commit()


@pytest.mark.asyncio
async def test_reads_the_dataset_backfilled_on_the_row(session_table):
    user_id, dataset_id = uuid4(), uuid4()
    session_table.append(("s-attributed", user_id))
    await ensure_and_touch_session(
        session_id="s-attributed", user_id=user_id, dataset_id=dataset_id
    )

    assert await get_session_dataset_id(user_id=user_id, session_id="s-attributed") == dataset_id


@pytest.mark.asyncio
async def test_falls_back_to_the_default_session_suffix(session_table):
    """Rows written before dataset threading carry the dataset only in their session_id."""
    user_id, dataset_id = uuid4(), uuid4()
    session_id = f"default_session_{dataset_id}"
    session_table.append((session_id, user_id))
    await ensure_and_touch_session(session_id=session_id, user_id=user_id)

    assert await get_session_dataset_id(user_id=user_id, session_id=session_id) == dataset_id


@pytest.mark.asyncio
async def test_unattributed_session_answers_none(session_table):
    user_id = uuid4()
    session_table.append(("s-plain", user_id))
    await ensure_and_touch_session(session_id="s-plain", user_id=user_id)

    assert await get_session_dataset_id(user_id=user_id, session_id="s-plain") is None


@pytest.mark.asyncio
async def test_missing_row_answers_none(session_table):
    assert await get_session_dataset_id(user_id=uuid4(), session_id="never-recorded") is None


@pytest.mark.asyncio
async def test_another_users_session_with_the_same_id_is_not_read(session_table):
    owner, stranger, dataset_id = uuid4(), uuid4(), uuid4()
    session_table.append(("shared-id", owner))
    await ensure_and_touch_session(session_id="shared-id", user_id=owner, dataset_id=dataset_id)

    assert await get_session_dataset_id(user_id=stranger, session_id="shared-id") is None
