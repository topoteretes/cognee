"""Backend capability tests for graph provenance on the default graph
backend (Ladybug/Kuzu) — COG-5522 Part 1.

These exercise the real adapter (declared STRING[] provenance columns + the
GraphMetadata marker table) against the Part 0 contract: source-ref
attach/remove invariants, the six lookups, delete-planning snapshots,
delete_edge_triples, the graph-provenance metadata marker, and belongs_to_set
detag. They mirror the in-memory FakeProvenanceGraphEngine semantics that
Part 2 is built against.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
    make_source_ref_key,
    make_source_run_ref,
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


class _Ent(DataPoint):
    """Minimal entity DataPoint with an indexed field and tag membership."""

    name: str
    metadata: dict = {"index_fields": ["name"]}


class _Structural(DataPoint):
    """Structural node that declares no index fields (e.g. a Timestamp)."""

    metadata: dict = {"index_fields": []}


def _new_adapter(tmp_path):
    return LadybugAdapter(str(tmp_path / "graph_db"))


async def _seed_two_entities(adapter):
    germany = _Ent(id=uuid4(), name="Germany")
    alice = _Ent(id=uuid4(), name="Alice")
    await adapter.add_nodes([germany, alice])
    return str(germany.id), str(alice.id)


async def test_graph_metadata_round_trip(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        # Unmarked graph returns an empty dict (NOT UnsupportedProvenanceCapability).
        assert await adapter.get_graph_metadata() == {}

        await adapter.set_graph_metadata(
            {
                GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
                GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
            }
        )
        meta = await adapter.get_graph_metadata()
        assert meta[GRAPH_DELETE_MODE_KEY] == GRAPH_DELETE_MODE_GRAPH_PROVENANCE
        assert meta[GRAPH_PROVENANCE_VERSION_KEY] == GRAPH_PROVENANCE_VERSION

        # A marked-but-data-empty graph still reads as empty (marker lives in a
        # separate table, is_empty is scoped to :Node).
        assert await adapter.is_empty() is True

        # set_graph_metadata is a merge/update, not a replace.
        await adapter.set_graph_metadata({"extra": "value"})
        meta = await adapter.get_graph_metadata()
        assert meta[GRAPH_DELETE_MODE_KEY] == GRAPH_DELETE_MODE_GRAPH_PROVENANCE
        assert meta["extra"] == "value"
    finally:
        await adapter.close()


async def test_attach_node_source_refs_materializes_all_fields(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        d1, d2 = uuid4(), uuid4()
        r1, r2 = uuid4(), uuid4()
        key_a = make_source_ref_key(d1, uuid4())
        key_b = make_source_ref_key(d2, uuid4())
        germany_id, _ = await _seed_two_entities(adapter)

        # Unset columns read back as empty lists.
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == []
        assert snap.source_dataset_ids == []
        assert snap.source_run_ids == []
        assert snap.source_run_refs == []

        await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == [key_a]
        assert snap.source_dataset_ids == [str(d1)]
        assert snap.source_run_ids == [str(r1)]
        assert snap.source_run_refs == [make_source_run_ref(r1, key_a)]

        # Second dataset/run set-merges into all four fields.
        await adapter.attach_node_source_refs([germany_id], [key_b], str(r2))
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert set(snap.source_ref_keys) == {key_a, key_b}
        assert snap.source_dataset_ids == sorted({str(d1), str(d2)})
        assert snap.source_run_ids == sorted({str(r1), str(r2)})

        # Re-attaching an existing (run, key) adds no duplicate run ref.
        await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert sorted(snap.source_run_refs) == sorted(
            {make_source_run_ref(r1, key_a), make_source_run_ref(r2, key_b)}
        )

        # Part 0 invariant: a NEW run re-touching an already-present key records
        # NO new run ref. Only the run that first attached the key owns it for
        # rollback, so rolling back the re-touching run cannot strip that
        # ownership (and delete an artifact a prior run still owns).
        await adapter.attach_node_source_refs([germany_id], [key_a], str(r2))
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert sorted(snap.source_run_refs) == sorted(
            {make_source_run_ref(r1, key_a), make_source_run_ref(r2, key_b)}
        )
        assert snap.source_run_ids == sorted({str(r1), str(r2)})
    finally:
        await adapter.close()


async def test_attach_without_pipeline_run_is_not_rollbackable_by_run(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1 = uuid4(), uuid4()
        key = make_source_ref_key(d1, uuid4())
        germany_id, _ = await _seed_two_entities(adapter)

        await adapter.attach_node_source_refs([germany_id], [key], None)
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == [key]
        assert snap.source_dataset_ids == [str(d1)]
        assert snap.source_run_ids == []
        assert snap.source_run_refs == []
        assert await adapter.find_node_source_refs_by_pipeline_run(str(r1)) == {}
    finally:
        await adapter.close()


async def test_remove_node_source_refs_updates_derived_and_is_idempotent(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        d1, d2, r1, r2 = uuid4(), uuid4(), uuid4(), uuid4()
        key_a = make_source_ref_key(d1, uuid4())
        key_b = make_source_ref_key(d2, uuid4())
        germany_id, _ = await _seed_two_entities(adapter)

        await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
        await adapter.attach_node_source_refs([germany_id], [key_b], str(r2))

        await adapter.remove_node_source_refs([germany_id], [key_a])
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == [key_b]
        assert snap.source_dataset_ids == [str(d2)]
        assert snap.source_run_ids == [str(r2)]
        assert snap.source_run_refs == [make_source_run_ref(r2, key_b)]
        # Removing one ref must not delete the artifact.
        assert await adapter.has_node(germany_id) is True

        # Idempotent: re-removing an already-removed ref and a missing node no-op.
        await adapter.remove_node_source_refs([germany_id], [key_a])
        await adapter.remove_node_source_refs(["does-not-exist"], [key_a])
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == [key_b]
    finally:
        await adapter.close()


async def test_node_lookups(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1 = uuid4(), uuid4()
        key_a = make_source_ref_key(d1, uuid4())
        key_b = make_source_ref_key(d1, uuid4())
        germany_id, alice_id = await _seed_two_entities(adapter)

        await adapter.attach_node_source_refs([germany_id], [key_a, key_b], str(r1))
        await adapter.attach_node_source_refs([alice_id], [key_a], str(r1))

        assert sorted(await adapter.find_nodes_by_source_ref(key_b)) == [germany_id]
        assert (
            await adapter.find_nodes_by_source_ref("source_ref:v1:%s:%s" % (uuid4(), uuid4())) == []
        )

        by_dataset = await adapter.find_node_source_refs_by_dataset(str(d1))
        assert set(by_dataset[germany_id]) == {key_a, key_b}
        assert by_dataset[alice_id] == [key_a]

        by_run = await adapter.find_node_source_refs_by_pipeline_run(str(r1))
        assert set(by_run[germany_id]) == {key_a, key_b}
        assert by_run[alice_id] == [key_a]
    finally:
        await adapter.close()


async def test_remove_keeps_dataset_id_when_sibling_ref_shares_dataset(tmp_path):
    """Removing one ref must KEEP its dataset id when another surviving ref
    still belongs to that dataset (guards cross-dataset-shared artifacts)."""
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1, r2 = uuid4(), uuid4(), uuid4()
        key_a = make_source_ref_key(d1, uuid4())  # same dataset d1
        key_b = make_source_ref_key(d1, uuid4())  # same dataset d1
        germany_id, alice_id = await _seed_two_entities(adapter)

        # Node: two refs from the SAME dataset, attached by different runs.
        await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
        await adapter.attach_node_source_refs([germany_id], [key_b], str(r2))
        await adapter.remove_node_source_refs([germany_id], [key_a])
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == [key_b]
        assert snap.source_dataset_ids == [str(d1)]  # d1 survives via key_b

        # Edge: same invariant.
        await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
        edge = EdgeIdentity(germany_id, alice_id, "knows")
        await adapter.attach_edge_source_refs([edge], [key_a], str(r1))
        await adapter.attach_edge_source_refs([edge], [key_b], str(r2))
        await adapter.remove_edge_source_refs([edge], [key_a])
        esnap = (await adapter.get_edge_delete_data([edge]))[edge]
        assert esnap.source_ref_keys == [key_b]
        assert esnap.source_dataset_ids == [str(d1)]
    finally:
        await adapter.close()


async def test_remove_keeps_run_id_when_sibling_ref_shares_run(tmp_path):
    """Removing one ref must KEEP its run id when another surviving ref was
    contributed by the same run (guards rollback candidate filtering)."""
    adapter = _new_adapter(tmp_path)
    try:
        d1, d2, r1 = uuid4(), uuid4(), uuid4()
        key_a = make_source_ref_key(d1, uuid4())
        key_b = make_source_ref_key(d2, uuid4())
        germany_id, alice_id = await _seed_two_entities(adapter)

        # Node: two refs attached by the SAME run r1.
        await adapter.attach_node_source_refs([germany_id], [key_a, key_b], str(r1))
        await adapter.remove_node_source_refs([germany_id], [key_a])
        snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert snap.source_ref_keys == [key_b]
        assert snap.source_run_ids == [str(r1)]  # r1 survives via key_b's run ref
        assert snap.source_run_refs == [make_source_run_ref(r1, key_b)]

        # Edge: same invariant.
        await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
        edge = EdgeIdentity(germany_id, alice_id, "knows")
        await adapter.attach_edge_source_refs([edge], [key_a, key_b], str(r1))
        await adapter.remove_edge_source_refs([edge], [key_a])
        esnap = (await adapter.get_edge_delete_data([edge]))[edge]
        assert esnap.source_ref_keys == [key_b]
        assert esnap.source_run_ids == [str(r1)]
        assert esnap.source_run_refs == [make_source_run_ref(r1, key_b)]
    finally:
        await adapter.close()


async def test_node_delete_data_snapshot_fields(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        germany = _Ent(id=uuid4(), name="Germany")
        ts = _Structural(id=uuid4())
        await adapter.add_nodes([germany, ts])
        germany_id, ts_id = str(germany.id), str(ts.id)

        snaps = await adapter.get_node_delete_data([germany_id, ts_id, "ghost"])
        # Missing artifacts are omitted, not surfaced as empty.
        assert set(snaps.keys()) == {germany_id, ts_id}
        assert snaps[germany_id].node_type == "_Ent"
        assert snaps[germany_id].indexed_fields == ["name"]
        assert snaps[germany_id].node_properties.get("name") == "Germany"
        # Structural node: empty indexed_fields.
        assert snaps[ts_id].indexed_fields == []

        assert await adapter.get_node_delete_data([]) == {}
    finally:
        await adapter.close()


async def test_edge_provenance_snapshot_and_lookups(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1 = uuid4(), uuid4()
        key = make_source_ref_key(d1, uuid4())
        germany_id, alice_id = await _seed_two_entities(adapter)

        await adapter.add_edges(
            [(germany_id, alice_id, "knows", {"edge_text": "Germany knows Alice"})]
        )
        await adapter.add_edges([(alice_id, germany_id, "likes", {})])
        edge = EdgeIdentity(germany_id, alice_id, "knows")
        edge_no_text = EdgeIdentity(alice_id, germany_id, "likes")

        await adapter.attach_edge_source_refs([edge], [key], str(r1))

        snap = (await adapter.get_edge_delete_data([edge]))[edge]
        assert snap.source_ref_keys == [key]
        assert snap.source_dataset_ids == [str(d1)]
        assert snap.edge_text == "Germany knows Alice"

        # Edge with no stored edge_text falls back to relationship_name.
        snap_fb = (await adapter.get_edge_delete_data([edge_no_text]))[edge_no_text]
        assert snap_fb.edge_text == "likes"

        assert await adapter.find_edges_by_source_ref(key) == [edge]
        assert await adapter.find_edge_source_refs_by_dataset(str(d1)) == {edge: [key]}
        assert await adapter.find_edge_source_refs_by_pipeline_run(str(r1)) == {edge: [key]}

        await adapter.remove_edge_source_refs([edge], [key])
        assert await adapter.find_edges_by_source_ref(key) == []
    finally:
        await adapter.close()


async def test_delete_edge_triples_preserves_endpoints(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        germany_id, alice_id = await _seed_two_entities(adapter)
        await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
        await adapter.add_edges([(germany_id, alice_id, "trusts", {"edge_text": "t2"})])

        await adapter.delete_edge_triples([EdgeIdentity(germany_id, alice_id, "knows")])

        # Only the matched relationship is gone; the other edge and both nodes survive.
        remaining = await adapter.get_edge_delete_data(
            [
                EdgeIdentity(germany_id, alice_id, "knows"),
                EdgeIdentity(germany_id, alice_id, "trusts"),
            ]
        )
        assert EdgeIdentity(germany_id, alice_id, "knows") not in remaining
        assert EdgeIdentity(germany_id, alice_id, "trusts") in remaining
        assert await adapter.has_node(germany_id) is True
        assert await adapter.has_node(alice_id) is True
    finally:
        await adapter.close()


async def test_remove_belongs_to_set_tags_scoped_and_unscoped(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        shared = _Ent(id=uuid4(), name="Shared", belongs_to_set=["Dev", "DevMirror"])
        same_tag = _Ent(id=uuid4(), name="SameTag", belongs_to_set=["Dev"])
        await adapter.add_nodes([shared, same_tag])
        shared_id, same_tag_id = str(shared.id), str(same_tag.id)

        # Scoped: only `shared` is detagged; `same_tag` keeps "Dev".
        await adapter.remove_belongs_to_set_tags(["Dev"], node_ids=[shared_id])
        s = (await adapter.get_node_delete_data([shared_id]))[shared_id]
        t = (await adapter.get_node_delete_data([same_tag_id]))[same_tag_id]
        assert s.node_properties.get("belongs_to_set") == ["DevMirror"]
        assert t.node_properties.get("belongs_to_set") == ["Dev"]

        # Unscoped removes only the requested tag everywhere.
        await adapter.remove_belongs_to_set_tags(["Dev"])
        t = (await adapter.get_node_delete_data([same_tag_id]))[same_tag_id]
        assert t.node_properties.get("belongs_to_set") == []

        # Empty tag list is a no-op.
        await adapter.remove_belongs_to_set_tags([])
        s = (await adapter.get_node_delete_data([shared_id]))[shared_id]
        assert s.node_properties.get("belongs_to_set") == ["DevMirror"]
    finally:
        await adapter.close()


async def test_rewrite_preserves_provenance(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1 = uuid4(), uuid4()
        key = make_source_ref_key(d1, uuid4())
        germany = _Ent(id=uuid4(), name="Germany")
        alice = _Ent(id=uuid4(), name="Alice")
        await adapter.add_nodes([germany, alice])
        germany_id, alice_id = str(germany.id), str(alice.id)

        await adapter.attach_node_source_refs([germany_id], [key], str(r1))
        await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
        edge = EdgeIdentity(germany_id, alice_id, "knows")
        await adapter.attach_edge_source_refs([edge], [key], str(r1))

        # Re-cognify: re-write the same node + edge with new content.
        await adapter.add_nodes([_Ent(id=germany.id, name="Germany UPDATED")])
        await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t2"})])

        node_snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
        assert node_snap.source_ref_keys == [key]
        assert node_snap.node_properties.get("name") == "Germany UPDATED"

        edge_snap = (await adapter.get_edge_delete_data([edge]))[edge]
        assert edge_snap.source_ref_keys == [key]
        assert edge_snap.edge_text == "t2"
    finally:
        await adapter.close()


# ----------------------------------------------------------------------------
# Folded provenance: stamping inside the add_nodes / add_edges write itself
# (COG-5522 #4/#8). The artifact is created and stamped in one atomic statement
# — no separate attach pass, so no write-then-attach window and no concurrent
# read-modify-write lost update.
# ----------------------------------------------------------------------------


async def test_add_nodes_folds_provenance_in_one_write(tmp_path):
    """A single folded add_nodes materializes all four provenance fields, so the
    node is findable by ref with nothing left to attach afterwards (closes #4)."""
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1 = uuid4(), uuid4()
        key = make_source_ref_key(d1, uuid4())
        node = _Ent(id=uuid4(), name="Germany")

        await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(r1))

        node_id = str(node.id)
        snap = (await adapter.get_node_delete_data([node_id]))[node_id]
        assert snap.source_ref_keys == [key]
        assert snap.source_dataset_ids == [str(d1)]
        assert snap.source_run_ids == [str(r1)]
        assert snap.source_run_refs == [make_source_run_ref(r1, key)]
        assert await adapter.find_nodes_by_source_ref(key) == [node_id]
    finally:
        await adapter.close()


async def test_add_edges_folds_provenance_in_one_write(tmp_path):
    """The edge variant stamps provenance in the same MERGE that writes the edge."""
    adapter = _new_adapter(tmp_path)
    try:
        d1, r1 = uuid4(), uuid4()
        key = make_source_ref_key(d1, uuid4())
        germany_id, alice_id = await _seed_two_entities(adapter)

        await adapter.add_edges(
            [(germany_id, alice_id, "knows", {"edge_text": "t"})],
            source_ref_key=key,
            pipeline_run_id=str(r1),
        )

        edge = EdgeIdentity(germany_id, alice_id, "knows")
        snap = (await adapter.get_edge_delete_data([edge]))[edge]
        assert snap.source_ref_keys == [key]
        assert snap.source_dataset_ids == [str(d1)]
        assert snap.source_run_ids == [str(r1)]
        assert snap.source_run_refs == [make_source_run_ref(r1, key)]
        assert await adapter.find_edges_by_source_ref(key) == [edge]
    finally:
        await adapter.close()


async def test_folded_attach_omitted_when_no_source_ref(tmp_path):
    """Without a source_ref_key the write stamps nothing — non-graph-provenance and
    other write sites are unaffected (back-compat)."""
    adapter = _new_adapter(tmp_path)
    try:
        node = _Ent(id=uuid4(), name="Germany")
        await adapter.add_nodes([node])  # no provenance args
        node_id = str(node.id)
        snap = (await adapter.get_node_delete_data([node_id]))[node_id]
        assert snap.source_ref_keys == []
        assert snap.source_run_refs == []
    finally:
        await adapter.close()


async def test_folded_attach_preserves_model_a_on_reattach(tmp_path):
    """Re-attaching an already-present key via a NEW run records no new run ref,
    even on the folded path — the in-statement guard reads pre-write ownership
    (Part 0 invariant preserved atomically)."""
    adapter = _new_adapter(tmp_path)
    try:
        key = make_source_ref_key(uuid4(), uuid4())
        r1, r2 = uuid4(), uuid4()
        node = _Ent(id=uuid4(), name="Shared")
        node_id = str(node.id)

        await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(r1))
        # Re-cognify the same data item in a new run: key already present.
        await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(r2))

        snap = (await adapter.get_node_delete_data([node_id]))[node_id]
        assert snap.source_ref_keys == [key]
        assert snap.source_run_refs == [make_source_run_ref(r1, key)]
        assert snap.source_run_ids == [str(r1)]
    finally:
        await adapter.close()


async def test_concurrent_folded_attach_keeps_all_keys(tmp_path):
    """The #8 regression: two concurrent folded writes attach DIFFERENT keys to
    the SAME node. With the old read-modify-write attach one key was lost; the
    folded single-statement set-merge keeps both."""
    adapter = _new_adapter(tmp_path)
    try:
        ds = uuid4()
        run = str(uuid4())
        key1 = make_source_ref_key(ds, uuid4())  # data item 1
        key2 = make_source_ref_key(ds, uuid4())  # data item 2
        node = _Ent(id=uuid4(), name="Alice")  # shared across both data items
        node_id = str(node.id)

        await asyncio.gather(
            adapter.add_nodes([node], source_ref_key=key1, pipeline_run_id=run),
            adapter.add_nodes([node], source_ref_key=key2, pipeline_run_id=run),
        )

        snap = (await adapter.get_node_delete_data([node_id]))[node_id]
        assert sorted(snap.source_ref_keys) == sorted([key1, key2]), (
            "concurrent folded stamps must not lose a source ref"
        )
        assert sorted(snap.source_run_refs) == sorted(
            [make_source_run_ref(UUID(run), key1), make_source_run_ref(UUID(run), key2)]
        )
    finally:
        await adapter.close()
