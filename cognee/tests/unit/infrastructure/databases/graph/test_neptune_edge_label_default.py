"""Regression test: Neptune get_predecessors/get_successors default edge label.

Both methods accept an optional ``edge_label`` and guard it with
``f" :{edge_label}" if edge_label is not None else ""``. The default used to be
the empty string ``""`` instead of ``None``. Since ``"" is not None`` is True,
calling either method without an edge label still took the "label" branch and
produced an invalid openCypher pattern (``<-[r :]-`` / ``-[r :]->``), which
Neptune rejects at query time. The Neo4j sibling defaults to ``None`` and works.

The methods are exercised unbound with a stub ``self`` (the adapter needs live
AWS config to instantiate), inspecting the generated query and return value.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB


@pytest.mark.asyncio
async def test_get_predecessors_without_edge_label_is_valid_opencypher():
    stub = SimpleNamespace(query=AsyncMock(return_value=[{"predecessor": "p1"}]))

    result = await NeptuneGraphDB.get_predecessors(stub, "n1")

    query = stub.query.call_args.args[0]
    # No label requested -> bare relationship, never the malformed "[r :]".
    assert "[r]" in query
    assert "[r :]" not in query
    assert result == ["p1"]


@pytest.mark.asyncio
async def test_get_successors_without_edge_label_is_valid_opencypher():
    stub = SimpleNamespace(query=AsyncMock(return_value=[{"successor": "s1"}]))

    result = await NeptuneGraphDB.get_successors(stub, "n1")

    query = stub.query.call_args.args[0]
    assert "[r]" in query
    assert "[r :]" not in query
    assert result == ["s1"]


@pytest.mark.asyncio
async def test_edge_label_still_applied_when_provided():
    stub = SimpleNamespace(query=AsyncMock(return_value=[]))

    await NeptuneGraphDB.get_predecessors(stub, "n1", edge_label="KNOWS")
    pred_query = stub.query.call_args.args[0]
    assert ":KNOWS" in pred_query

    await NeptuneGraphDB.get_successors(stub, "n1", edge_label="KNOWS")
    succ_query = stub.query.call_args.args[0]
    assert ":KNOWS" in succ_query
