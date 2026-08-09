"""build_brains_payload: the overview across every dataset a user may read.

Patches at the ``fetch_dataset_graph_data`` boundary rather than the graph
engine: seed resolution and bounded truncation are already exercised end to
end in test_visualize_subgraph.py, and get_all_user_permission_datasets is
CLO-399's existing group-aware union, not new here. What is new, and what
this pins, is the loop and keying logic in build_brains_payload itself —
one independent fetch per authorized dataset, assembled into a dict keyed by
dataset id with the {"name", "nodes", "links", "node_set_colors"} shape, not
the full visualize payload.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Same shadowing gotcha as test_visualize_subgraph.py: cognee.api.v1.__init__
# rebinds `visualize` on the v1 package to the visualize_graph *function*, so a
# dotted-string patch target has to go through sys.modules instead.
from cognee.api.v1.visualize.visualize import build_brains_payload as _build_brains_payload  # noqa: F401,E501

visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]


def _dataset(dataset_id: str, name: str):
    return SimpleNamespace(id=dataset_id, name=name, owner_id="owner-1")


def _graph_for(label: str):
    return (
        [(f"{label}-n1", {"type": "Entity", "name": label})],
        [],
    )


@pytest.mark.asyncio
async def test_each_dataset_gets_its_own_fetch_keyed_by_id():
    datasets = [_dataset("D1", "billing"), _dataset("D2", "support")]
    fetched_for = []

    async def fake_fetch(dataset, **_kwargs):
        fetched_for.append(dataset.id)
        return _graph_for(dataset.name)

    with (
        patch.object(
            visualize_module,
            "get_all_user_permission_datasets",
            AsyncMock(return_value=datasets),
        ),
        patch.object(visualize_module, "fetch_dataset_graph_data", fake_fetch),
    ):
        payload = await visualize_module.build_brains_payload(user=MagicMock())

    assert set(payload) == {"D1", "D2"}
    assert payload["D1"]["name"] == "billing"
    assert payload["D2"]["name"] == "support"
    assert set(payload["D1"]) == {"name", "nodes", "links", "node_set_colors"}
    # One fetch per dataset, not a single fetch reused for both — each
    # dataset's data lives behind its own dataset-scoped read.
    assert fetched_for == ["D1", "D2"]


@pytest.mark.asyncio
async def test_no_datasets_is_an_empty_payload_not_an_error():
    with patch.object(
        visualize_module, "get_all_user_permission_datasets", AsyncMock(return_value=[])
    ):
        payload = await visualize_module.build_brains_payload(user=MagicMock())

    assert payload == {}
