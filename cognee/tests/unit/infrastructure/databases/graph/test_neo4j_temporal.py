from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.tasks.temporal_graph.models import Timestamp


@pytest.mark.asyncio
async def test_collect_time_ids_returns_list_for_unwind_params():
    """collect_time_ids returns a list, matching the ladybug adapter.

    It previously returned a pre-quoted, comma-joined string built for string
    interpolation into the Cypher text.
    """
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"id": "timestamp-1"}, {"id": "timestamp-2"}])

    ids = await adapter.collect_time_ids(time_to=Timestamp(year=1980))

    assert ids == ["timestamp-1", "timestamp-2"]


@pytest.mark.asyncio
async def test_collect_events_binds_ids_as_list():
    """A List[str] must reach Cypher as a bound parameter, not as its repr.

    `collect_events` declared `ids: List[str]` but interpolated it with
    `.format(quoted=ids)`, so a real list rendered its Python repr and produced
    `UNWIND [['a', 'b']] AS uid`. That binds uid to the whole list, so
    `MATCH (start {id: uid})` matches nothing and the query returns zero events
    without raising.
    """
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[])

    await adapter.collect_events(ids=["timestamp-1", "timestamp-2"])

    cypher, params = adapter.query.await_args.args
    assert params == {"ids": ["timestamp-1", "timestamp-2"]}
    assert "$ids" in cypher, "ids must be bound as a parameter, not interpolated"
    assert "[['timestamp-1'" not in cypher, "the list must not be rendered as a nested list"


@pytest.mark.asyncio
async def test_collect_events_accepts_legacy_quoted_id_string():
    """The pre-quoted string form still works, as it does in the ladybug adapter."""
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[])

    await adapter.collect_events(ids="'timestamp-1', 'timestamp-2'")

    _, params = adapter.query.await_args.args
    assert params == {"ids": ["timestamp-1", "timestamp-2"]}


@pytest.mark.asyncio
async def test_collect_time_ids_output_feeds_collect_events():
    """The two halves must compose: whatever one returns, the other must accept.

    This is the path temporal_retriever takes -- collect_time_ids straight into
    collect_events -- and the shape mismatch between them was the bug.
    """
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"id": "timestamp-1"}, {"id": "timestamp-2"}])

    ids = await adapter.collect_time_ids(
        time_from=Timestamp(year=1980), time_to=Timestamp(year=1990)
    )

    adapter.query = AsyncMock(return_value=[])
    await adapter.collect_events(ids=ids)

    _, params = adapter.query.await_args.args
    assert params == {"ids": ["timestamp-1", "timestamp-2"]}
