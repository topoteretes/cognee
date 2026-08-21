"""Unit tests for the store's portable fallbacks — no graph database, no LLM.

``update_node`` and ``delete_edge_triples`` are optional ``GraphDBInterface``
extensions (the base raises), so the store must keep working on adapters that
lack them:

- ``upsert_preference_node`` falls back to the full-node ``add_nodes`` upsert
  when ``update_node`` raises ``NotImplementedError``.
- ``delete_prefers_edges`` neutralizes the edges via ``add_edges`` when
  ``delete_edge_triples`` raises ``UnsupportedProvenanceCapability`` (or
  ``NotImplementedError``).

Plus the read side: ``load_preference_state`` traverses only ``prefers``
edges (cheap read) and still returns a node that has no such edges.
"""

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.modules.user_preferences import store as store_module
from cognee.modules.user_preferences.constants import (
    NEUTRAL_WEIGHT,
    PREFERS_RELATIONSHIP,
)
from cognee.modules.user_preferences.store import (
    delete_prefers_edges,
    load_preference_state,
    preference_node_id,
    upsert_preference_node,
)


class FakeGraphEngine:
    def __init__(self, *, update_node_result=None, delete_edge_triples_error=None):
        self.update_node_result = update_node_result
        self.delete_edge_triples_error = delete_edge_triples_error
        self.update_node_calls = []
        self.add_nodes_calls = []
        self.add_edges_calls = []
        self.delete_edge_triples_calls = []

    async def update_node(self, node_id, values):
        self.update_node_calls.append((node_id, values))
        if self.update_node_result is None:
            raise NotImplementedError("update_node is not implemented for this adapter")
        return self.update_node_result

    async def add_nodes(self, nodes):
        self.add_nodes_calls.append(list(nodes))

    async def add_edges(self, edges):
        self.add_edges_calls.append(list(edges))

    async def delete_edge_triples(self, edges):
        self.delete_edge_triples_calls.append(list(edges))
        if self.delete_edge_triples_error is not None:
            raise self.delete_edge_triples_error


def _wire(monkeypatch, engine):
    async def fake_get_graph_engine():
        return engine

    monkeypatch.setattr(store_module, "get_graph_engine", fake_get_graph_engine)


class TestLoadPreferenceState:
    class _NeighborhoodEngine:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges
            self.calls = []

        async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
            self.calls.append({"node_ids": node_ids, "depth": depth, "edge_types": edge_types})
            return self.nodes, self.edges

    @pytest.mark.asyncio
    async def test_read_traverses_only_prefers_edges(self, monkeypatch):
        # The read must stay cheap: an unfiltered depth-1 neighborhood drags in
        # every neighbour with all its properties (full liked-chunk texts, the
        # NodeSet and through it every other preference node) just to pull two
        # numbers off each edge.
        pref_id = str(preference_node_id("user-1", "ds-1"))
        engine = self._NeighborhoodEngine(
            nodes=[(pref_id, {"text": "likes graphs", "turn_counter": 2})],
            edges=[
                (
                    pref_id,
                    "chunk-1",
                    PREFERS_RELATIONSHIP,
                    {"weight": 0.8, "updated_at_turn": 2},
                ),
                # Edges between collected nodes still come back unfiltered;
                # only prefers edges out of the seed may land in the result.
                ("chunk-1", "chunk-2", "related_to", {"weight": 0.9}),
            ],
        )
        _wire(monkeypatch, engine)

        node, stored = await load_preference_state("user-1", "ds-1")

        assert engine.calls == [
            {"node_ids": [pref_id], "depth": 1, "edge_types": [PREFERS_RELATIONSHIP]}
        ]
        assert node == {"text": "likes graphs", "turn_counter": 2}
        assert stored == {"chunk-1": {"weight": 0.8, "updated_at_turn": 2}}

    @pytest.mark.asyncio
    async def test_node_with_no_prefers_edges_still_reads_back(self, monkeypatch):
        # Adapters return seed nodes even when the edge filter matches nothing,
        # so a preference node holding only stated text must not read as absent.
        pref_id = str(preference_node_id("user-1", "ds-1"))
        engine = self._NeighborhoodEngine(
            nodes=[(pref_id, {"text": "prefers concise answers", "turn_counter": 0})],
            edges=[],
        )
        _wire(monkeypatch, engine)

        node, stored = await load_preference_state("user-1", "ds-1")

        assert node == {"text": "prefers concise answers", "turn_counter": 0}
        assert stored == {}


class TestUpsertPreferenceNodeFallback:
    @pytest.mark.asyncio
    async def test_update_node_success_skips_full_write(self, monkeypatch):
        engine = FakeGraphEngine(update_node_result=True)
        _wire(monkeypatch, engine)

        pref_id = await upsert_preference_node("user-1", "ds-1", text="t", turn_counter=3)

        assert pref_id == str(preference_node_id("user-1", "ds-1"))
        assert engine.update_node_calls[0][1]["turn_counter"] == 3
        assert engine.add_nodes_calls == []
        assert engine.add_edges_calls == []

    @pytest.mark.asyncio
    async def test_update_node_miss_creates_node(self, monkeypatch):
        engine = FakeGraphEngine(update_node_result=False)
        _wire(monkeypatch, engine)

        pref_id = await upsert_preference_node("user-1", "ds-1", text="t")

        assert len(engine.add_nodes_calls) == 1
        assert len(engine.add_edges_calls) == 1
        written_ids = {str(node.id) for node in engine.add_nodes_calls[0]}
        assert pref_id in written_ids

    @pytest.mark.asyncio
    async def test_update_node_not_implemented_falls_back_to_full_write(self, monkeypatch):
        # Non-Ladybug adapters raise NotImplementedError; the write must land
        # through add_nodes (a true upsert everywhere) instead of failing.
        engine = FakeGraphEngine(update_node_result=None)
        _wire(monkeypatch, engine)

        pref_id = await upsert_preference_node(
            "user-1", "ds-1", text="likes graphs", turn_counter=7, text_watermark="w"
        )

        assert pref_id == str(preference_node_id("user-1", "ds-1"))
        assert len(engine.add_nodes_calls) == 1
        preference = next(node for node in engine.add_nodes_calls[0] if str(node.id) == pref_id)
        assert preference.text == "likes graphs"
        assert preference.turn_counter == 7
        assert preference.text_watermark == "w"
        # NodeSet membership still written so the node stays listable/prunable.
        assert len(engine.add_edges_calls) == 1


class TestDeletePrefersEdgesFallback:
    @pytest.mark.asyncio
    async def test_delete_supported_deletes_and_never_rewrites(self, monkeypatch):
        engine = FakeGraphEngine()
        _wire(monkeypatch, engine)

        await delete_prefers_edges("user-1", "ds-1", ["target-1", "target-2"])

        assert len(engine.delete_edge_triples_calls) == 1
        assert engine.add_edges_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [UnsupportedProvenanceCapability(), NotImplementedError("nope")],
        ids=["unsupported-provenance", "not-implemented"],
    )
    async def test_delete_unsupported_neutralizes_edges(self, monkeypatch, error):
        engine = FakeGraphEngine(delete_edge_triples_error=error)
        _wire(monkeypatch, engine)

        await delete_prefers_edges("user-1", "ds-1", ["target-1", "target-2"])

        assert len(engine.add_edges_calls) == 1
        edges = engine.add_edges_calls[0]
        assert {edge[1] for edge in edges} == {"target-1", "target-2"}
        for source_id, _target_id, rel, props in edges:
            assert source_id == str(preference_node_id("user-1", "ds-1"))
            assert rel == PREFERS_RELATIONSHIP
            assert props["weight"] == NEUTRAL_WEIGHT
            assert props["updated_at_turn"] == 0

    @pytest.mark.asyncio
    async def test_empty_targets_touch_nothing(self, monkeypatch):
        engine = FakeGraphEngine(delete_edge_triples_error=NotImplementedError())
        _wire(monkeypatch, engine)

        await delete_prefers_edges("user-1", "ds-1", [])

        assert engine.delete_edge_triples_calls == []
        assert engine.add_edges_calls == []
