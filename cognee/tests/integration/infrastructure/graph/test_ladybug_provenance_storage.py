"""Ladybug/Kuzu-specific graph-provenance storage-shape tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.engine import DataPoint

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
    name: str
    metadata: dict = {"index_fields": ["name"]}


def _new_adapter(tmp_path):
    return LadybugAdapter(str(tmp_path / "graph_db"))


async def _seed_two_entities(adapter):
    germany = _Ent(id=uuid4(), name="Germany")
    alice = _Ent(id=uuid4(), name="Alice")
    await adapter.add_nodes([germany, alice])
    return str(germany.id), str(alice.id)


async def _raw_node_source_ref_keys(adapter, node_id: str):
    rows = await adapter.query(
        "MATCH (n:Node) WHERE n.id = $id RETURN n.source_ref_keys",
        {"id": node_id},
    )
    return rows[0][0]


async def _raw_edge_source_ref_keys(adapter, edge: EdgeIdentity):
    rows = await adapter.query(
        """
        MATCH (a:Node)-[r:EDGE]->(b:Node)
        WHERE a.id = $source_id AND b.id = $target_id AND r.relationship_name = $rel
        RETURN r.source_ref_keys
        """,
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "rel": edge.relationship_name,
        },
    )
    return rows[0][0]


async def test_folded_node_provenance_uses_delimited_scalar_storage(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        key = make_source_ref_key(uuid4(), uuid4())
        node = _Ent(id=uuid4(), name="Germany")

        await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(uuid4()))

        assert await _raw_node_source_ref_keys(adapter, str(node.id)) == f"|{key}|"
    finally:
        await adapter.close()


async def test_folded_edge_provenance_uses_delimited_scalar_storage(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        key = make_source_ref_key(uuid4(), uuid4())
        germany_id, alice_id = await _seed_two_entities(adapter)

        await adapter.add_edges(
            [(germany_id, alice_id, "knows", {"edge_text": "t"})],
            source_ref_key=key,
            pipeline_run_id=str(uuid4()),
        )

        edge = EdgeIdentity(germany_id, alice_id, "knows")
        assert await _raw_edge_source_ref_keys(adapter, edge) == f"|{key}|"
    finally:
        await adapter.close()


async def test_folded_edge_multiple_owners_use_one_delimited_scalar(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        key_a = make_source_ref_key(uuid4(), uuid4())
        key_b = make_source_ref_key(uuid4(), uuid4())
        germany_id, alice_id = await _seed_two_entities(adapter)
        edge = EdgeIdentity(germany_id, alice_id, "knows")

        await adapter.add_edges(
            [(germany_id, alice_id, "knows", {"edge_text": "t"})],
            source_ref_key=key_a,
            pipeline_run_id=str(uuid4()),
        )
        await adapter.add_edges(
            [(germany_id, alice_id, "knows", {"edge_text": "t2"})],
            source_ref_key=key_b,
            pipeline_run_id=str(uuid4()),
        )

        assert await _raw_edge_source_ref_keys(adapter, edge) == f"|{key_a}|{key_b}|"
    finally:
        await adapter.close()


async def test_delimited_source_ref_filter_excludes_lookalike_raw_token(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        node = _Ent(id=uuid4(), name="Shared")
        await adapter.add_nodes([node])
        node_id = str(node.id)
        short_ref = "source_ref:v1:prefix"
        lookalike_ref = "source_ref:v1:prefix-extra"

        await adapter._write_node_provenance(
            [
                {
                    "id": node_id,
                    "refs": [lookalike_ref],
                    "datasets": [],
                    "runs": [],
                    "run_refs": [],
                }
            ]
        )

        assert await adapter.find_nodes_by_source_ref(short_ref) == []
        assert await adapter.find_nodes_by_source_ref(lookalike_ref) == [node_id]
    finally:
        await adapter.close()


async def test_long_delimited_source_refs_round_trip_and_filter(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        dataset_id = uuid4()
        keys = [make_source_ref_key(dataset_id, uuid4()) for _ in range(256)]
        node = _Ent(id=uuid4(), name="Shared")
        await adapter.add_nodes([node])
        node_id = str(node.id)

        await adapter._write_node_provenance(
            [
                {
                    "id": node_id,
                    "refs": keys,
                    "datasets": [str(dataset_id)],
                    "runs": [],
                    "run_refs": [],
                }
            ]
        )

        snap = (await adapter.get_node_delete_data([node_id]))[node_id]
        assert snap.source_ref_keys == keys
        assert await adapter.find_nodes_by_source_ref(keys[128]) == [node_id]
    finally:
        await adapter.close()
