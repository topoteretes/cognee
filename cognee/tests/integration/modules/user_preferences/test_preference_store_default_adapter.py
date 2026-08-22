"""Store tests against the REAL default graph adapter (Ladybug/Kuzu embedded).

The unit suite replaces the graph engine with fakes, so nothing there proves
the one seam the whole isolation promise rests on: that the ``is_internal``
marker survives a real write and comes back inside the properties that
``get_graph_data()`` returns — that property is what every read chokepoint
checks to keep preference nodes out of retrieval output. These tests are
offline (no LLM, no embeddings): they drive the real adapter directly, the
same way the graph-provenance default-stack tests do.
"""

from __future__ import annotations

import pytest

from cognee.infrastructure.engine import is_internal_node
from cognee.modules.user_preferences import store as store_module
from cognee.modules.user_preferences.constants import PREFERS_RELATIONSHIP
from cognee.modules.user_preferences.models import UserPreference
from cognee.modules.user_preferences.store import (
    load_preference_state,
    preference_node_id,
    upsert_preference_node,
    write_prefers_edges,
)

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LADYBUG, reason="ladybug not installed"),
]


async def test_is_internal_survives_real_write_and_get_graph_data(tmp_path):
    """Write a UserPreference through the real adapter; read it back via
    ``get_graph_data()`` and assert ``is_internal`` is present in the returned
    properties — the flag every retrieval chokepoint filters on."""
    adapter = LadybugAdapter(str(tmp_path / "graph_db"))
    try:
        preference = UserPreference(
            id=preference_node_id("user-1", "ds-1"),
            user_id="user-1",
            dataset_id="ds-1",
            text="prefers concise answers",
            turn_counter=3,
        )
        await adapter.add_nodes([preference])

        nodes, _edges = await adapter.get_graph_data()
        properties = next(
            (props for node_id, props in nodes if str(node_id) == str(preference.id)), None
        )

        assert properties is not None, "preference node did not read back from the real adapter"
        assert properties.get("is_internal") is True
        assert is_internal_node(properties)
        # The rest of the payload survives the same properties round trip.
        assert properties.get("text") == "prefers concise answers"
        assert properties.get("turn_counter") == 3
    finally:
        await adapter.close()


async def test_store_roundtrip_on_real_adapter(tmp_path, monkeypatch):
    """Drive the store's own write/read paths (``upsert_preference_node``,
    ``write_prefers_edges``, ``load_preference_state``) against the real
    adapter instead of a fake, so ``update_node`` + ``get_neighborhood``
    are exercised for real."""
    adapter = LadybugAdapter(str(tmp_path / "graph_db"))

    async def real_get_graph_engine():
        return adapter

    monkeypatch.setattr(store_module, "get_graph_engine", real_get_graph_engine)

    try:
        pref_id = await upsert_preference_node(
            "user-1", "ds-1", text="likes graphs", turn_counter=2, text_watermark="w1"
        )

        # A content node the prefers edge can point at.
        target = UserPreference(
            id=preference_node_id("user-1", "ds-target"),
            user_id="user-1",
            dataset_id="ds-target",
        )
        await adapter.add_nodes([target])
        await write_prefers_edges("user-1", "ds-1", {str(target.id): 0.8}, updated_at_turn=2)

        node, stored = await load_preference_state("user-1", "ds-1")

        assert pref_id == str(preference_node_id("user-1", "ds-1"))
        assert node is not None
        assert node.get("is_internal") is True
        assert node.get("text") == "likes graphs"
        assert node.get("turn_counter") == 2
        assert stored == {str(target.id): {"weight": 0.8, "updated_at_turn": 2}}

        # Second upsert takes the in-place update path on Ladybug and must
        # not lose the marker.
        await upsert_preference_node("user-1", "ds-1", text="likes graphs a lot", turn_counter=5)
        node, _stored = await load_preference_state("user-1", "ds-1")
        assert node.get("is_internal") is True
        assert node.get("text") == "likes graphs a lot"
        assert node.get("turn_counter") == 5

        # The prefers edge reads back through get_graph_data too.
        _nodes, edges = await adapter.get_graph_data()
        prefers = [
            (source, target_id, props)
            for source, target_id, rel, props in edges
            if rel == PREFERS_RELATIONSHIP
        ]
        assert [(source, target_id) for source, target_id, _props in prefers] == [
            (pref_id, str(target.id))
        ]
        assert prefers[0][2].get("weight") == 0.8
    finally:
        await adapter.close()
