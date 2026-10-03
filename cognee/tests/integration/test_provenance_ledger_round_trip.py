"""Integration: the audit ledger across a full memory lifecycle.

cognify writes rows -> forget(memory_only) tombstones them -> re-cognify
resurrects what is re-extracted -> an external anchor is taken -> the dataset
is deleted and every live row is retracted. The hash chain, referential
integrity, and the anchor must hold at every step, and the export must be
re-verifiable from its own content.

Needs an LLM key (the graph is extracted by the configured model).
"""

import json
import os
import pathlib
from collections import Counter

import pytest

import cognee
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.engine.operations.setup import setup
from cognee.modules.provenance import anchors, get_provenance_manager
from cognee.modules.provenance.integrity import canonical_entity_id, compute_checksum
from cognee.modules.provenance.models import ProvenanceEntry
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

    # 1. cognify writes rows -------------------------------------------------- #
    await cognee.add([TEXT], dataset_name=dataset_name, user=user)
    await cognee.cognify([dataset_name], user=user)
    dataset_id = (await get_datasets_by_name(dataset_name, user.id))[0].id

    stats = await manager.get_statistics(dataset_id=dataset_id)
    assert stats["live_entries"] > 0
    assert stats["invalidated_count"] == 0
    assert {"entity", "relationship"} <= set(stats["entity_types"])
    await _assert_intact(manager, dataset_id)
    live_before, dead_before = await _live_and_dead(manager, dataset_id)
    assert live_before and not dead_before

    # Lineage of an edge reaches its endpoints, with checksums verified.
    edge_key = next(key for key in live_before if key.startswith("rel:"))
    lineage = await manager.get_lineage(edge_key)
    assert lineage["entity_count"] >= 2
    assert lineage["integrity_verified"] is True

    # 2. forget(memory_only) tombstones everything ---------------------------- #
    await cognee.forget(dataset=dataset_name, memory_only=True, user=user)
    live, dead = await _live_and_dead(manager, dataset_id)
    assert not live
    assert dead == live_before
    tombstone = await manager.get_provenance(edge_key)
    assert tombstone["invalidated"] is True
    assert tombstone["invalidation_reason"] == "dataset_deleted"
    await _assert_intact(manager, dataset_id)

    # 3. re-cognify resurrects what is extracted again ----------------------- #
    await cognee.cognify([dataset_name], user=user)
    live, dead = await _live_and_dead(manager, dataset_id)
    assert live, "re-cognify must bring rows back to life"
    assert not (live & dead)
    resurrected = next(iter(live & live_before), None)
    if resurrected is not None:
        history = await manager.revision_history(resurrected)
        kinds = Counter("tombstone" if v.get("invalidated") else "assertion" for v in history)
        assert kinds["tombstone"] >= 1 and kinds["assertion"] >= 2
        assert history[-1].get("invalidated") is not True
    await _assert_intact(manager, dataset_id)

    # 4. anchor the chain head ----------------------------------------------- #
    anchor = await manager.anchor()
    assert anchor is not None and anchors.signature_valid(anchor)
    assert (await manager.verify_anchors())["valid"] is True

    # 5. delete the dataset: live rows retracted, nothing removed ------------ #
    total_before = (await manager.get_statistics())["total_entries"]
    await cognee.forget(dataset=dataset_name, user=user)
    live, dead = await _live_and_dead(manager, dataset_id)
    assert not live and dead
    assert (await manager.get_statistics())["total_entries"] > total_before  # tombstones add rows
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
