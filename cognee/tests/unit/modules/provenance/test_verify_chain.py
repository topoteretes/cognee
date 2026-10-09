"""Hash-chain verification: tamper detection, deletion detection, concurrency."""

import asyncio

import pytest

from cognee.modules.provenance import storage

pytestmark = pytest.mark.asyncio


async def _seed(manager, count=5):
    for index in range(1, count + 1):
        await manager.track_entity(f"e{index}", source=f"doc-{index}")


class TestVerifyChain:
    async def test_clean_chain_is_valid(self, manager):
        await _seed(manager, 3)
        await manager.track_entity("e1", source="doc-1b")  # re-track: archive + relabel
        await manager.track_chunk("chunk-1", source_document="doc-1")
        await manager.track_relationship("rel:e1:knows:e2", source="chunk-1")
        await manager.invalidate("e2", agent_id="auditor", reason="retracted")

        report = await manager.verify_chain()
        assert report["valid"] is True
        assert report["broken_links"] == []

    async def test_in_place_edit_is_checksum_mismatch(self, manager, tamper):
        await _seed(manager, 3)
        await tamper("UPDATE provenance_entries SET agent_id = 'evil' WHERE entity_id = 'e2'")

        report = await manager.verify_chain()
        assert report["valid"] is False
        assert len(report["broken_links"]) == 1
        assert report["broken_links"][0] == {
            "entity_id": "e2",
            "sequence_id": 2,
            "reason": "checksum_mismatch",
        }

    async def test_deleting_a_middle_row_is_a_chain_break(self, manager, tamper):
        await _seed(manager, 5)
        await tamper("DELETE FROM provenance_entries WHERE sequence_id = 2")

        report = await manager.verify_chain()
        assert report["valid"] is False
        # Exactly the successor of the deleted row is flagged — no cascade.
        assert len(report["broken_links"]) == 1
        broken = report["broken_links"][0]
        assert broken["reason"] == "chain_break"
        assert broken["sequence_id"] == 3
        assert broken["expected_sequence_id"] == 2
        assert broken["actual_previous_checksum"] != broken["expected_previous_checksum"]

    async def test_streaming_pagination_matches_full_scan(self, manager):
        await _seed(manager, 5)
        streamed = [entry async for entry in storage.iter_chained(page_size=2)]
        assert [entry.sequence_id for entry in streamed] == [1, 2, 3, 4, 5]
        everything = [entry async for entry in storage.iter_all(page_size=2)]
        assert len(everything) == 5

    async def test_concurrent_appends_never_fork_the_chain(self, manager):
        await asyncio.gather(
            *(manager.track_entity(f"e{index}", source=f"doc-{index}") for index in range(20))
        )

        entries = await storage.retrieve_all()
        assert {entry.sequence_id for entry in entries} == set(range(1, 21))
        assert (await manager.verify_chain())["valid"] is True
