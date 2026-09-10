"""collect_session_events: dataset scoping of the auto-discovered sessions.

Pins the predicate COG-6121 is about — a dataset's timeline shows that
dataset's activity, not every dataset the caller has ever queried — at the
level the filter actually runs: the lifecycle listing query, with its
ORDER BY / LIMIT intact.

The relational engine is stubbed with an in-memory SQLite one so the SQL
predicate itself is exercised rather than mocked away; that is the part
test_live_events.py cannot see, since it replaces collect_session_events
wholesale.
"""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.session_lifecycle.metrics import SessionStatus
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.visualization import session_events

# Must come from importlib, not `import ... as`: the package __init__ binds
# the name `get_session_manager` to the *function*, so both the dotted
# string patch target and `import a.b.c as x` resolve to that function
# instead of the module, unless something imported the submodule first.
# That import-order dependence is why this passed locally and failed on CI.
get_session_manager_module = importlib.import_module(
    "cognee.infrastructure.session.get_session_manager"
)

DATASET_A = UUID("aaaaaaaa-1111-4111-8111-1111111111ab")
DATASET_B = UUID("bbbbbbbb-2222-4222-8222-2222222222cd")
USER_ID = UUID("cccccccc-3333-4333-8333-3333333333ef")
USER = SimpleNamespace(id=USER_ID)

BASE_TIME = datetime(2026, 8, 3, 9, 0, 0, tzinfo=timezone.utc)

# All-decimal UUIDs are avoided on purpose: the column renders as UUID,
# which matches none of SQLite's affinity rules and so falls through to
# NUMERIC, and a UUID made only of digits then comes back as a float.


@pytest_asyncio.fixture
async def seed():
    """An in-memory lifecycle table, plus a writer for seeding rows."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all, tables=[SessionRecord.__table__])
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def add(session_id, dataset_id, *, age_seconds=0, user_id=USER_ID):
        async with maker() as db:
            db.add(
                SessionRecord(
                    session_id=session_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    status=SessionStatus.RUNNING.value,
                    started_at=BASE_TIME,
                    last_activity_at=BASE_TIME - timedelta(seconds=age_seconds),
                )
            )
            await db.commit()

    yield SimpleNamespace(add=add, engine=SimpleNamespace(get_async_session=maker))
    await engine.dispose()


def _patch_engine(stub):
    return patch(
        "cognee.infrastructure.databases.relational.get_relational_engine",
        return_value=stub.engine,
    )


async def _listed(stub, dataset_id, limit=session_events.MAX_SESSIONS_SCANNED):
    with _patch_engine(stub):
        return await session_events._list_recent_session_ids(USER_ID, limit, dataset_id=dataset_id)


@pytest.mark.asyncio
async def test_session_attributed_to_b_is_excluded_for_a_and_included_for_b(seed):
    await seed.add("session-a", DATASET_A)
    await seed.add("session-b", DATASET_B)

    assert await _listed(seed, DATASET_A) == ["session-a"]
    assert await _listed(seed, DATASET_B) == ["session-b"]


@pytest.mark.asyncio
async def test_null_dataset_is_excluded_unless_the_session_id_carries_the_suffix(seed):
    await seed.add("opaque-session", None)
    await seed.add(f"default_session_{DATASET_A}", None)

    listed = await _listed(seed, DATASET_A)

    assert listed == [f"default_session_{DATASET_A}"]
    assert "opaque-session" not in listed


@pytest.mark.asyncio
async def test_limit_applies_after_scoping_so_the_newest_for_that_dataset_survive(seed):
    """A global limit then filter would drop these: B is newer throughout."""
    for index in range(session_events.MAX_SESSIONS_SCANNED + 2):
        await seed.add(f"b-{index}", DATASET_B, age_seconds=index)
    for index in range(3):
        await seed.add(f"a-{index}", DATASET_A, age_seconds=100 + index)

    listed = await _listed(seed, DATASET_A)

    assert listed == ["a-0", "a-1", "a-2"]


@pytest.mark.asyncio
async def test_another_users_session_on_the_same_dataset_is_never_listed(seed):
    """The suffix clause must not widen the listing past the owner."""
    await seed.add(f"default_session_{DATASET_A}", DATASET_A, user_id=uuid4())
    await seed.add("mine", DATASET_A)

    assert await _listed(seed, DATASET_A) == ["mine"]


@pytest.mark.asyncio
async def test_no_dataset_id_keeps_the_unscoped_listing(seed):
    await seed.add("session-a", DATASET_A)
    await seed.add("opaque-session", None)

    assert sorted(await _listed(seed, None)) == ["opaque-session", "session-a"]


@pytest.mark.asyncio
async def test_explicit_session_ids_bypass_the_dataset_filter():
    """An explicit list is an intentional override, so scoping must not apply."""

    async def _get_session(*, user_id, session_id):
        return [
            SimpleNamespace(
                qa_id=session_id,
                time="2026-08-03T09:00:00.000000",
                question=f"q-{session_id}",
                answer="a",
                used_graph_element_ids={"node_ids": ["n1"], "edge_ids": []},
                feedback_score=None,
                feedback_text=None,
                memify_metadata=None,
            )
        ]

    with (
        patch.object(
            get_session_manager_module,
            "get_session_manager",
            return_value=SimpleNamespace(is_available=True, get_session=_get_session),
        ),
        patch.object(session_events, "_list_recent_session_ids") as listing,
    ):
        events = await session_events.collect_session_events(
            user=USER, session_ids=["session-b"], dataset_id=DATASET_A
        )

    listing.assert_not_called()
    assert [event["question"] for event in events] == ["q-session-b"]
