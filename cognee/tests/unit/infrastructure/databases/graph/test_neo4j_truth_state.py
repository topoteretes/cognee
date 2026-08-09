"""Neo4j truth-state methods.

Before these, Neo4j inherited the interface's NotImplementedError defaults for
truth state, so the truth build skipped Neo4j entirely (feedback weights were
already implemented separately).
"""

from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _adapter(rows):
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=rows)
    return adapter


@pytest.mark.asyncio
async def test_get_node_truth_state_parses_rows():
    adapter = _adapter(
        [
            {"id": "n1", "truth_alignment": [0.1, 0.2], "truth_epoch": 3},
            {"id": "n2", "truth_alignment": None, "truth_epoch": "bad"},
        ]
    )

    state = await adapter.get_node_truth_state(["n1", "n2", ""])

    assert state["n1"] == {"truth_alignment": [0.1, 0.2], "truth_epoch": 3}
    assert state["n2"] == {"truth_alignment": [], "truth_epoch": None}


@pytest.mark.asyncio
async def test_set_node_truth_state_reports_per_id_success():
    adapter = _adapter([{"id": "n1"}])

    result = await adapter.set_node_truth_state(
        {
            "n1": {"truth_alignment": [1.0, 0.0], "truth_epoch": 2},
            "n-missing": {"truth_alignment": [0.5], "truth_epoch": 2},
        }
    )

    assert result == {"n1": True, "n-missing": False}
    updates = adapter.query.await_args.args[1]["updates"]
    assert {"id": "n1", "truth_alignment": [1.0, 0.0], "truth_epoch": 2} in updates


@pytest.mark.asyncio
async def test_empty_inputs_are_noops():
    adapter = _adapter([])

    assert await adapter.get_node_truth_state([]) == {}
    assert await adapter.set_node_truth_state({}) == {}
    adapter.query.assert_not_awaited()
