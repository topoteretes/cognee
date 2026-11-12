"""Integration: the audit ledger across a full memory lifecycle.

cognify records the data item (and nothing it produced) -> forget(memory_only)
keeps the document, so the ledger row stays live -> re-cognify of the same
content is not a new version -> an external anchor is taken -> the dataset is
deleted and the data item is retracted. The hash chain, referential integrity,
and the anchor must hold at every step, and the export must be re-verifiable
from its own content.

Needs an LLM key (the graph is extracted by the configured model).
"""

import json
import os
import pathlib

import pytest

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.engine.operations.setup import setup
from cognee.modules.provenance import anchors, get_provenance_manager
from cognee.modules.provenance.integrity import canonical_entity_id, compute_checksum
from cognee.modules.provenance.models import ProvenanceEntry
from cognee.modules.provenance.tombstones import ledger_data_item_key
from cognee.modules.users.methods import get_default_user

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.getenv("LLM_API_KEY"), reason="needs an LLM key"),
]

TEXT = (
    "Marie Curie was a physicist and chemist who conducted pioneering research on "
    "radioactivity. She was born in Warsaw and later moved to Paris, where she worked "
    "with her husband Pierre Curie at the University of Paris."
)


async def _assert_intact(manager, dataset_id):
    assert (await manager.verify_chain())["valid"] is True
    scoped = await manager.verify_chain(dataset_id=dataset_id)
    assert scoped["valid"] is True
    assert (await manager.check(dataset_id=dataset_id))["valid"] is True


async def _live_and_dead(manager, dataset_id) -> tuple[set[str], set[str]]:
    live, dead = set(), set()
    async for row in manager.export(dataset_id=dataset_id, include_archived=False):
        (dead if row["invalidated"] else live).add(row["entity_id"])
    return live, dead


async def test_ledger_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVENANCE_TRACKING", "true")
    monkeypatch.setenv("PROVENANCE_ANCHOR_KEY", "integration-test-key")
    monkeypatch.setenv("PROVENANCE_ANCHOR_PATH", str(tmp_path / "anchors.jsonl"))
    get_cognify_config.cache_clear()

    base = pathlib.Path(__file__).parent.parent
    cognee.config.data_root_directory(str(base / ".data_storage/test_provenance_ledger"))
    cognee.config.system_root_directory(str(base / ".cognee_system/test_provenance_ledger"))
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()
    manager = get_provenance_manager()
    dataset_name = "ledger_round_trip"

    # 1. cognify records the data item, and only the data item ---------------- #
    await cognee.add([TEXT], dataset_name=dataset_name, user=user)
    await cognee.cognify([dataset_name], user=user)
    dataset_id = (await get_datasets_by_name(dataset_name, user.id))[0].id
    [data] = await cognee.datasets.list_data(dataset_id, user)
    item_key = ledger_data_item_key(dataset_id, data.id)

    stats = await manager.get_statistics(dataset_id=dataset_id)
    assert stats["entity_types"] == {"data_item": 1}
    assert stats["live_entries"] == 1 and stats["invalidated_count"] == 0
    row = await manager.get_provenance(item_key)
    assert row["metadata"]["snapshot"]["fields"] == {"content_hash": data.content_hash}
    assert row["agent_id"] == str(user.id)
    async with set_database_global_context_variables(dataset_id, user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
    assert len(nodes) > 1 and edges, "the graph holds what the ledger does not repeat"
    await _assert_intact(manager, dataset_id)

    # 2. forget(memory_only) keeps the document: no tombstone ---------------- #
    await cognee.forget(dataset=dataset_name, memory_only=True, user=user)
    live, dead = await _live_and_dead(manager, dataset_id)
    assert (live, dead) == ({item_key}, set())
    assert [row.id for row in await cognee.datasets.list_data(dataset_id, user)] == [data.id]
    await _assert_intact(manager, dataset_id)

    # 3. re-cognify of the same content is not a new version ------------------ #
    await cognee.cognify([dataset_name], user=user)
    live, dead = await _live_and_dead(manager, dataset_id)
    assert (live, dead) == ({item_key}, set())
    assert len(await manager.revision_history(item_key)) == 1
    await _assert_intact(manager, dataset_id)

    # 4. anchor the chain head ----------------------------------------------- #
    anchor = await manager.anchor()
    assert anchor is not None and anchors.signature_valid(anchor)
    assert (await manager.verify_anchors())["valid"] is True

    # 5. delete the dataset: the data item is retracted, nothing removed ----- #
    total_before = (await manager.get_statistics())["total_entries"]
    await cognee.forget(dataset=dataset_name, user=user)
    live, dead = await _live_and_dead(manager, dataset_id)
    assert (live, dead) == (set(), {item_key})
    tombstone = await manager.get_provenance(item_key)
    assert tombstone["invalidation_reason"] == "dataset_deleted"
    assert tombstone["invalidated_by"] == str(user.id)
    assert (await manager.get_statistics())["total_entries"] == total_before + 1  # one tombstone
    await _assert_intact(manager, dataset_id)
    assert (await manager.verify_anchors())["valid"] is True  # anchored row still there

    # 6. export re-verifies offline ------------------------------------------ #
    rows = [row async for row in manager.export()]
    assert [r["sequence_id"] for r in rows] == list(range(1, len(rows) + 1))
    previous = None
    for row in rows:
        entry = ProvenanceEntry(**json.loads(json.dumps(row, default=str)))
        assert compute_checksum(entry) == row["checksum"], canonical_entity_id(entry)
        assert row["previous_checksum"] == previous
        previous = row["checksum"]
