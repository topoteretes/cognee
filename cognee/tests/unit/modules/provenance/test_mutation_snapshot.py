"""Mutation snapshots: what a row records, deltas, no-op suppression, drift."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.provenance import drift, snapshot
from cognee.modules.provenance.snapshot import (
    EXCLUDED_FIELDS,
    MAX_INLINE_CHARS,
    build_snapshot,
    content_hash,
    diff_snapshots,
    snapshot_fields,
    snapshot_from_mapping,
    snapshot_metadata,
)
from cognee.modules.provenance.tombstones import ledger_edge_key, ledger_node_key

pytestmark = pytest.mark.asyncio


class Place(DataPoint):
    name: str
    description: str | None = None
    tags: list[str] = []
    population: int | None = None
    metadata: dict = {"index_fields": ["name"]}


class Person(DataPoint):
    name: str
    lives_in: Place | None = None
    friends: list["Person"] = []
    metadata: dict = {"index_fields": ["name"]}


class TestSnapshotFields:
    def test_excludes_bookkeeping_and_nested_data_points(self):
        place = Place(
            name="Paris", description="capital", tags=["eu", "city"], population=2_000_000
        )
        person = Person(name="Ada", lives_in=place, friends=[Person(name="Bob")])
        fields = snapshot_fields(person)
        assert fields["name"] == "Ada"
        assert fields["type"] == "Person"
        assert "lives_in" not in fields and "friends" not in fields
        assert not (set(fields) & EXCLUDED_FIELDS)
        assert "feedback_weight" not in fields  # tuned by improve(), not content

    def test_long_text_is_hashed_not_stored(self):
        long = "x" * (MAX_INLINE_CHARS + 1)
        fields = snapshot_fields(Place(name="P", description=long))
        assert set(fields["description"]) == {"sha256", "len"}
        assert fields["description"]["len"] == MAX_INLINE_CHARS + 1
        assert snapshot_fields(Place(name="P", description="short"))["description"] == "short"

    def test_hash_is_deterministic_and_ignores_timestamps_and_ids(self):
        a = Place(name="Paris", tags=["b", "a"])
        b = Place(name="Paris", tags=["a", "b"], id=uuid4())
        b.created_at = a.created_at + 1000
        assert build_snapshot(a)["hash"] == build_snapshot(b)["hash"]
        assert build_snapshot(Place(name="Rome"))["hash"] != build_snapshot(a)["hash"]

    def test_non_models_get_no_snapshot(self):
        assert build_snapshot(SimpleNamespace(name="x")) is None
        assert snapshot_metadata(SimpleNamespace(name="x")) == {}
        assert "snapshot" in snapshot_metadata(Place(name="x"))

    def test_graph_node_round_trip_hashes_identically(self):
        place = Place(name="Paris", description="capital", tags=["eu"], population=5)
        recorded = build_snapshot(place)
        # What an adapter hands back: JSON-flattened properties, lists as JSON strings.
        node = json.loads(json.dumps(place.model_dump(), default=str))
        node["tags"] = json.dumps(node["tags"])
        node["_label"] = "Node"
        observed = snapshot_from_mapping(node, recorded["fields"])
        assert content_hash(observed) == recorded["hash"]

        node["description"] = "the capital"
        drifted = snapshot_from_mapping(node, recorded["fields"])
        assert diff_snapshots(recorded["fields"], drifted) == {
            "description": ["capital", "the capital"]
        }

    def test_diff(self):
        assert diff_snapshots({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == {
            "b": [2, 3],
            "c": [None, 4],
        }
        assert diff_snapshots(None, None) == {}


class TestVersioningWithSnapshots:
    async def test_same_content_same_document_is_a_noop(self, manager):
        ds, ref = uuid4(), "source_ref:v1:a:b"
        key = ledger_node_key(ds, "paris")
        meta = {"name": "Paris", **snapshot_metadata(Place(name="Paris", description="capital"))}
        first = await manager.track_entity(key, source="c1", metadata=meta, source_ref_key=ref)
        again = await manager.track_entity(key, source="c2", metadata=meta, source_ref_key=ref)
        assert again.sequence_id == first.sequence_id
        assert len(await manager.revision_history(key)) == 1
        assert (await manager.get_statistics())["total_entries"] == 1

    async def test_same_content_other_document_versions(self, manager):
        ds = uuid4()
        key = ledger_node_key(ds, "paris")
        meta = {"name": "Paris", **snapshot_metadata(Place(name="Paris"))}
        await manager.track_entity(key, source="c1", metadata=meta, source_ref_key="doc-a")
        second = await manager.track_entity(key, source="c9", metadata=meta, source_ref_key="doc-b")
        history = await manager.revision_history(key)
        assert len(history) == 2
        assert history[-1]["delta"] == {}
        assert second.source_ref_key == "doc-b"

    async def test_changed_content_versions_with_delta(self, manager):
        ds, ref = uuid4(), "doc-a"
        key = ledger_node_key(ds, "paris")
        v1 = snapshot_metadata(Place(name="Paris", description="capital", population=1))
        v2 = snapshot_metadata(Place(name="Paris", description="capital of France", population=2))
        await manager.track_entity(key, source="c1", metadata={**v1}, source_ref_key=ref)
        await manager.track_entity(key, source="c1", metadata={**v2}, source_ref_key=ref)
        history = await manager.revision_history(key)
        assert [v["version"] for v in history] == [1, 2]
        assert "delta" not in history[0]
        assert history[1]["delta"] == {
            "description": ["capital", "capital of France"],
            "population": [1, 2],
        }
        assert history[1]["content_hash"] == v2["snapshot"]["hash"]
        assert (await manager.verify_chain())["valid"] is True

    async def test_explicit_revision_intent_is_never_a_noop(self, manager):
        key = ledger_node_key(uuid4(), "paris")
        meta = snapshot_metadata(Place(name="Paris"))
        await manager.track_entity(key, source="c1", metadata=meta, source_ref_key="d")
        await manager.track_entity(
            key, source="c1", metadata=meta, source_ref_key="d", revision_type="correction"
        )
        assert len(await manager.revision_history(key)) == 2

    async def test_tombstone_then_same_content_resurrects(self, manager):
        key = ledger_node_key(uuid4(), "paris")
        meta = snapshot_metadata(Place(name="Paris"))
        await manager.track_entity(key, source="c1", metadata=meta, source_ref_key="d")
        await manager.invalidate(key, "auditor", reason="gone")
        live = await manager.track_entity(key, source="c1", metadata=meta, source_ref_key="d")
        assert live.invalidated is False
        history = await manager.revision_history(key)
        assert [v.get("invalidated", False) for v in history] == [False, True, False]
        assert "delta" not in history[1]  # the tombstone carries no delta
        assert history[2]["delta"] == {}

    async def test_without_snapshots_every_retrack_versions(self, manager):
        key = ledger_node_key(uuid4(), "paris")
        await manager.track_entity(key, source="c1", metadata={"name": "Paris"}, source_ref_key="d")
        await manager.track_entity(key, source="c1", metadata={"name": "Paris"}, source_ref_key="d")
        assert len(await manager.revision_history(key)) == 2


class TestDrift:
    @pytest.fixture
    def graph(self, monkeypatch):
        engine = SimpleNamespace(nodes={}, get_nodes=None)

        async def get_nodes(node_ids):
            return [engine.nodes[i] for i in node_ids if i in engine.nodes]

        engine.get_nodes = get_nodes
        monkeypatch.setattr(drift, "get_graph_engine", AsyncMock(return_value=engine))

        class _Ctx:
            def __init__(self, *args, **kwargs):
                self.args = args

            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return False

        monkeypatch.setattr(drift, "set_database_global_context_variables", _Ctx)
        return engine

    async def _seed(self, manager, graph, ds):
        paris = Place(name="Paris", description="capital", tags=["eu"])
        rome = Place(name="Rome", description="eternal")
        for place in (paris, rome):
            graph.nodes[str(place.id)] = json.loads(json.dumps(place.model_dump(), default=str))
            await manager.track_entity(
                ledger_node_key(ds, str(place.id)),
                source="c1",
                metadata={"name": place.name, **snapshot_metadata(place)},
                source_ref_key="d",
            )
        await manager.track_relationship(
            ledger_edge_key(ds, str(paris.id), str(rome.id), "near"), source="c1"
        )
        await manager.track_entity(
            ledger_node_key(ds, "legacy"), source="c1", metadata={"name": "L"}
        )
        return paris, rome

    async def test_matching_graph_is_valid(self, manager, graph):
        ds = uuid4()
        await self._seed(manager, graph, ds)
        result = await manager.check_drift(ds, uuid4())
        assert result["valid"] is True
        assert result["checked"] == 2  # relationship + legacy row skipped
        assert result["unsnapshotted"] == 1
        assert result["drifted"] == [] and result["missing_in_graph"] == []

    async def test_out_of_band_edit_and_delete_are_reported(self, manager, graph):
        ds = uuid4()
        paris, rome = await self._seed(manager, graph, ds)
        graph.nodes[str(paris.id)]["description"] = "forged"
        del graph.nodes[str(rome.id)]

        result = await manager.check_drift(ds, uuid4())
        assert result["valid"] is False
        assert result["missing_in_graph"] == [ledger_node_key(ds, str(rome.id))]
        [item] = result["drifted"]
        assert item["entity_id"] == ledger_node_key(ds, str(paris.id))
        assert item["delta"] == {"description": ["capital", "forged"]}

    async def test_tombstoned_rows_are_not_checked(self, manager, graph):
        ds = uuid4()
        _, rome = await self._seed(manager, graph, ds)
        del graph.nodes[str(rome.id)]
        await manager.invalidate(ledger_node_key(ds, str(rome.id)), "forget", reason="deleted")
        result = await manager.check_drift(ds, uuid4())
        assert result["valid"] is True and result["checked"] == 1

    async def test_dataset_context_is_entered(self, manager, graph, monkeypatch):
        seen = {}

        class _Ctx:
            def __init__(self, dataset_id, owner_id=None):
                seen["args"] = (dataset_id, owner_id)

            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return False

        monkeypatch.setattr(drift, "set_database_global_context_variables", _Ctx)
        ds, owner = uuid4(), uuid4()
        await manager.check_drift(str(ds), str(owner))
        assert seen["args"] == (ds, owner)
        assert isinstance(seen["args"][0], UUID)
