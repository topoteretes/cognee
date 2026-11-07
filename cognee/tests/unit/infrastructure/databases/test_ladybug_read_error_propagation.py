"""Ladybug reads must not turn a store failure into an empty graph.

#4348 fixed this for ``has_edges``: it used to ``except Exception: return []``, so a
corrupt or unavailable store read as "none of these edges exist". The same adapter's
single and batch reads kept the pattern - ``get_node`` and ``extract_node`` answered
``None``, and the list reads answered ``[]`` - so a failed query still looked like a
node or a neighbourhood that is not there. They now re-raise after logging, matching
``has_edges`` and the other backends (neo4j re-raises; postgres/turso let it propagate).

The adapter is instantiated via ``__new__`` to bypass the real database connection,
as in test_has_edges_error_propagation.py.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

READS = [
    ("extract_node", ("n1",)),
    ("extract_nodes", (["n1", "n2"],)),
    ("get_node", ("n1",)),
    ("get_nodes", (["n1", "n2"],)),
    ("get_edges", ("n1",)),
    ("get_neighbors", ("n1",)),
    ("get_predecessors", ("n1",)),
    ("get_successors", ("n1",)),
    ("get_connections", ("n1",)),
]


def _adapter_with_query(query_mock) -> LadybugAdapter:
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = query_mock
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("method, args", READS)
async def test_read_propagates_store_failure(method, args):
    """A query failure must raise, not be swallowed into None or an empty result."""
    adapter = _adapter_with_query(AsyncMock(side_effect=RuntimeError("WAL corrupt")))
    with pytest.raises(RuntimeError, match="WAL corrupt"):
        await getattr(adapter, method)(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize("method, args", READS)
async def test_read_with_no_rows_is_still_empty(method, args):
    """A query that ran and matched nothing is a real empty answer and stays one."""
    adapter = _adapter_with_query(AsyncMock(return_value=[]))
    result = await getattr(adapter, method)(*args)
    assert result in (None, [])
