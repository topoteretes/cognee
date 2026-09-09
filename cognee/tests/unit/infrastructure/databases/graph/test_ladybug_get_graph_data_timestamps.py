"""Regression test for #3529.

On the Ladybug backend, a feedback-weight (or truth-state) write must not drop ``created_at`` /
``updated_at`` from what ``get_graph_data`` returns.

Before the fix, ``_build_node_feedback_updates`` / ``_build_node_truth_state_updates`` rebuilt the node
``properties`` blob *excluding* ``created_at`` / ``updated_at``, while ``get_graph_data`` reads timestamps
only from that blob — so they came back ``None`` after any feedback/truth-state write.
"""

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.engine import DataPoint


class _Claim(DataPoint):
    name: str = ""
    created_at: int = 0
    updated_at: int = 0
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_get_graph_data_preserves_timestamps_after_feedback_write(tmp_path):
    adapter = LadybugAdapter(str(tmp_path / "kuzu_ts"))
    node = _Claim(name="A", created_at=1000, updated_at=2000)
    await adapter.add_nodes([node])

    # present right after ingest
    nodes, _ = await adapter.get_graph_data()
    props = {nid: p for nid, p in nodes}[str(node.id)]
    assert props.get("created_at") == 1000
    assert props.get("updated_at") == 2000

    # a feedback-weight write must NOT drop them (this is the #3529 regression)
    await adapter.set_node_feedback_weights({str(node.id): 0.7})

    nodes, _ = await adapter.get_graph_data()
    props = {nid: p for nid, p in nodes}[str(node.id)]
    assert props.get("created_at") == 1000  # was None before the fix
    assert props.get("updated_at") == 2000  # was None before the fix
    assert props.get("feedback_weight") == 0.7
