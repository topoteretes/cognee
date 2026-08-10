"""``has_edges`` must not turn a store failure into a wrong answer.

The Ladybug adapter's batch existence check used to ``except Exception: return
[]``. Callers (the cognify dedup in ``retrieve_existing_edges``) read ``[]`` as
"none of these edges exist" and go on to write them, so when the underlying
store is unavailable/corrupt the run reports success while persisting nothing
(issue #4348). A failed existence check must surface the error, matching the
other backends (neo4j re-raises; postgres/turso let it propagate).

The adapter is instantiated via ``__new__`` to bypass the real database
connection — ``has_edges`` only depends on ``self.query``.
"""

import pytest
from unittest.mock import AsyncMock

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


def _adapter_with_query(query_mock) -> LadybugAdapter:
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = query_mock
    return adapter


@pytest.mark.asyncio
async def test_has_edges_returns_existing_edges_on_success():
    adapter = _adapter_with_query(AsyncMock(return_value=[("n1", "n2", "rel")]))
    assert await adapter.has_edges([("n1", "n2", "rel")]) == [("n1", "n2", "rel")]


@pytest.mark.asyncio
async def test_has_edges_propagates_store_failure():
    """A query failure must raise, not be swallowed into an empty result."""
    adapter = _adapter_with_query(AsyncMock(side_effect=RuntimeError("WAL corrupt")))
    with pytest.raises(RuntimeError, match="WAL corrupt"):
        await adapter.has_edges([("n1", "n2", "rel")])


@pytest.mark.asyncio
async def test_has_edges_empty_input_short_circuits():
    """Empty input is a real empty answer and must not touch the store."""
    query = AsyncMock()
    adapter = _adapter_with_query(query)
    assert await adapter.has_edges([]) == []
    query.assert_not_awaited()
