"""ProvenanceManager behavior over an isolated async SQLite ledger."""

import pytest

from cognee.modules.provenance import storage
from cognee.modules.provenance.integrity import compute_checksum

pytestmark = pytest.mark.asyncio


async def _archive_entries(entity_id):
    return [
        entry
        for entry in await storage.retrieve_all()
        if entry.entity_id.startswith(f"{entity_id}:v:")
    ]


class TestTrackEntity:
    async def test_create(self, manager):
        entry = await manager.track_entity("e1", source="/tmp/doc.txt", metadata={"name": "A"})
        assert entry is not None
        assert entry.sequence_id == 1
        assert entry.previous_checksum is None
        assert entry.checksum == compute_checksum(entry)
        assert entry.first_seen == entry.last_updated == entry.timestamp

        stored = await manager.get_provenance("e1")
        assert stored["source_document"] == "/tmp/doc.txt"
        assert stored["metadata"] == {"name": "A"}

    async def test_validation(self, manager):
        with pytest.raises(ValueError):
            await manager.track_entity(None, source="s")
        with pytest.raises(TypeError):
            await manager.track_entity(123, source="s")

    async def test_retrack_archives_old_version(self, manager):
        first = await manager.track_entity("e1", source="doc-a")
        second = await manager.track_entity("e1", source="doc-b")

        archives = await _archive_entries("e1")
        assert len(archives) == 1
        archive = archives[0]
        # Pure relabel: the archive keeps the OLD checksum/sequence/prev link.
        assert archive.checksum == first.checksum
        assert archive.sequence_id == first.sequence_id
        assert archive.previous_checksum == first.previous_checksum
        # And, since entity_id is excluded from the hash, it still verifies.
        assert compute_checksum(archive) == archive.checksum

        assert second.previous_version_id == archive.entity_id
        assert second.first_seen == first.first_seen
        assert second.sequence_id == 2
        assert second.previous_checksum == first.checksum
        # No explicit parent: the archival link doubles as the parent, but is
        # NOT a derivation.
        assert second.parent_entity_id == archive.entity_id
        assert second.derived_from_id is None

    async def test_parent_precedence_kwarg_wins(self, manager):
        await manager.track_entity("a", source="root")
        await manager.track_entity("b", source="root")
        entry = await manager.track_entity(
            "c", source="b", metadata={"derived_from": "b"}, parent_entity_id="a"
        )
        assert entry.parent_entity_id == "a"
        assert entry.derived_from_id == "a"

    async def test_parent_precedence_metadata_derived_from(self, manager):
        await manager.track_entity("a", source="root")
        entry = await manager.track_entity("c", source="untracked", metadata={"derived_from": "a"})
        assert entry.parent_entity_id == "a"
        assert entry.derived_from_id == "a"

    async def test_parent_precedence_source_heuristic(self, manager):
        await manager.track_entity("a", source="root")
        entry = await manager.track_entity("c", source="a")
        assert entry.parent_entity_id == "a"
        assert entry.derived_from_id == "a"

    async def test_untracked_source_gives_no_parent(self, manager):
        entry = await manager.track_entity("c", source="nowhere.txt")
        assert entry.parent_entity_id is None
        assert entry.derived_from_id is None

    async def test_archive_plus_explicit_parent_lands_in_used_entities(self, manager):
        await manager.track_entity("a", source="root")
        await manager.track_entity("e1", source="doc-a")
        entry = await manager.track_entity("e1", source="doc-b", parent_entity_id="a")
        archives = await _archive_entries("e1")
        assert entry.derived_from_id == "a"
        assert entry.previous_version_id == archives[0].entity_id
        assert archives[0].entity_id in entry.used_entities


class TestTrackChunkAndRelationship:
    async def test_track_chunk_sets_parent_and_derived_from(self, manager):
        await manager.track_entity("doc", source="/tmp/doc.txt", entity_type="document")
        entry = await manager.track_chunk(
            "chunk-1",
            source_document="/tmp/doc.txt",
            parent_chunk_id="doc",
            start_index=0,
            end_index=0,
            chunk_size=128,
        )
        assert entry.entity_type == "chunk"
        assert entry.activity_id == "chunking"
        assert entry.parent_entity_id == "doc"
        assert entry.derived_from_id == "doc"
        assert entry.metadata == {"chunk_size": 128}

    async def test_relationship_never_versions(self, manager):
        first = await manager.track_relationship("rel:a:knows:b", source="chunk-1")
        again = await manager.track_relationship("rel:a:knows:b", source="chunk-2")

        assert first.entity_type == "relationship"
        assert first.activity_id == "relationship_tracking"
        # Re-track is a chain-preserving no-op returning the stored entry.
        assert again.checksum == first.checksum
        assert again.sequence_id == first.sequence_id
        assert again.previous_version_id is None
        assert await _archive_entries("rel:a:knows:b") == []
        assert (await manager.get_statistics())["entity_types"]["relationship"] == 1

    async def test_relationship_records_used_entities(self, manager):
        entry = await manager.track_relationship(
            "rel:a:knows:b", source="chunk-1", used_entities=["a", "b"]
        )
        assert entry.used_entities == ["a", "b"]


class TestInvalidate:
    async def test_invalidate_tombstone(self, manager):
        await manager.track_entity("e1", source="doc-a")
        entry = await manager.invalidate("e1", agent_id="auditor", reason="retracted")

        assert entry.invalidated is True
        assert entry.invalidated_by == "auditor"
        assert entry.invalidation_reason == "retracted"
        assert entry.checksum == compute_checksum(entry)

        stored = await manager.get_provenance("e1")
        assert stored["invalidated"] is True

        archives = await _archive_entries("e1")
        assert len(archives) == 1
        assert archives[0].invalidated is False
        assert entry.previous_version_id == archives[0].entity_id

        # The tombstone is dated when the invalidation happened, not when the
        # fact was first tracked.
        assert entry.timestamp == entry.last_updated == entry.invalidated_at_time
        assert entry.timestamp >= archives[0].timestamp
        assert entry.first_seen == archives[0].first_seen
        history = await manager.revision_history("e1")
        assert history[-1]["recorded_at"] == entry.timestamp
        assert history[0]["valid_until"] == entry.timestamp

        assert (await manager.verify_chain())["valid"] is True

    async def test_invalidate_untracked_raises(self, manager):
        with pytest.raises(ValueError):
            await manager.invalidate("ghost", agent_id="auditor")


class TestRevisionHistory:
    async def test_order_and_synthesized_validity(self, manager):
        await manager.track_entity("e1", source="v1")
        await manager.track_entity("e1", source="v2")
        await manager.track_entity("e1", source="v3")

        history = await manager.revision_history("e1")
        assert [item["version"] for item in history] == [1, 2, 3]
        recorded = [item["recorded_at"] for item in history]
        assert recorded == sorted(recorded)
        # Every non-current version's window closes at the next version.
        assert history[0]["valid_until"] == history[1]["recorded_at"]
        assert history[1]["valid_until"] == history[2]["recorded_at"]
        assert history[2]["valid_until"] is None

    async def test_cycle_guard(self, manager, tamper):
        await manager.track_entity("e1", source="v1")
        await manager.track_entity("e1", source="v2")
        archives = await _archive_entries("e1")
        # Forge a cycle: archive points back at the live entry.
        await tamper(
            "UPDATE provenance_entries SET previous_version_id = 'e1' WHERE entity_id = :aid",
            {"aid": archives[0].entity_id},
        )
        history = await manager.revision_history("e1")
        assert len(history) == 2  # terminated, no infinite walk

    async def test_untracked_is_empty(self, manager):
        assert await manager.revision_history("ghost") == []


class TestLineage:
    async def test_get_lineage_aggregation(self, manager):
        await manager.track_entity("doc", source="/tmp/doc.txt", metadata={"kind": "pdf"})
        await manager.track_chunk("chunk-1", source_document="/tmp/doc.txt", parent_chunk_id="doc")
        await manager.track_entity(
            "ent-1", source="chunk-1", metadata={"kind": "entity", "name": "Alice"}
        )

        lineage = await manager.get_lineage("ent-1")
        assert lineage["entity_id"] == "ent-1"
        assert lineage["entity_count"] == 3
        assert [item["entity_id"] for item in lineage["lineage_chain"]] == [
            "ent-1",
            "chunk-1",
            "doc",
        ]
        assert set(lineage["source_documents"]) == {"chunk-1", "/tmp/doc.txt"}
        # Ancestors first, queried entity wins on conflicting keys.
        assert lineage["metadata"]["kind"] == "entity"
        assert lineage["metadata"]["name"] == "Alice"
        assert lineage["integrity_verified"] is True

    async def test_get_lineage_untracked(self, manager):
        assert await manager.get_lineage("ghost") == {}

    async def test_integrity_flag_flips_on_tamper(self, manager, tamper):
        await manager.track_entity("doc", source="/tmp/doc.txt")
        await manager.track_chunk("chunk-1", source_document="/tmp/doc.txt", parent_chunk_id="doc")
        await tamper("UPDATE provenance_entries SET agent_id = 'evil' WHERE entity_id = 'doc'")
        lineage = await manager.get_lineage("chunk-1")
        assert lineage["integrity_verified"] is False

    async def test_trace_lineage_max_depth(self, manager):
        await manager.track_entity("doc", source="/tmp/doc.txt")
        await manager.track_chunk("chunk-1", source_document="/tmp/doc.txt", parent_chunk_id="doc")
        await manager.track_entity("ent-1", source="chunk-1")
        entries = await manager.trace_lineage("ent-1", max_depth=1)
        assert [entry.entity_id for entry in entries] == ["ent-1"]


class TestBatch:
    async def test_batch_commits_one_chained_transaction(self, manager, monkeypatch):
        calls = []
        original = storage.append_chained_many

        async def counting(write_fns):
            calls.append(len(write_fns))
            return await original(write_fns)

        monkeypatch.setattr(storage, "append_chained_many", counting)

        batch = manager.batch()
        batch.track_entity("doc", source="/tmp/doc.txt", entity_type="document")
        batch.track_chunk("chunk-1", source_document="/tmp/doc.txt", parent_chunk_id="doc")
        batch.track_entity("ent-1", source="chunk-1")
        batch.track_relationship(
            "rel:chunk-1:contains:ent-1", source="chunk-1", used_entities=["chunk-1", "ent-1"]
        )
        results = await batch.commit()

        assert calls == [4]  # one append_chained_many for the whole batch
        assert [entry.sequence_id for entry in results] == [1, 2, 3, 4]
        # Later operations see earlier ones inside the same transaction: the
        # source-heuristic parent resolved against the just-written chunk.
        assert results[2].parent_entity_id == "chunk-1"
        assert (await manager.verify_chain())["valid"] is True
        assert len(batch) == 0  # commit drains the batch

    async def test_batch_noop_retrack_consumes_no_slot(self, manager):
        await manager.track_relationship("rel:a:knows:b", source="chunk-1")
        batch = manager.batch()
        batch.track_relationship("rel:a:knows:b", source="chunk-2")
        batch.track_entity("e1", source="doc")
        results = await batch.commit()
        assert results[0].sequence_id == 1  # the stored entry — no new slot
        assert results[1].sequence_id == 2  # chain advanced only for the real write
        assert (await manager.verify_chain())["valid"] is True

    async def test_batch_validation_raises_at_add_time(self, manager):
        batch = manager.batch()
        with pytest.raises(ValueError):
            batch.track_entity(None, source="doc")
        with pytest.raises(TypeError):
            batch.track_relationship(123, source="doc")

    async def test_batch_commit_degrades_gracefully(self, manager, monkeypatch):
        async def explode(write_fns):
            raise RuntimeError("db down")

        monkeypatch.setattr(storage, "append_chained_many", explode)
        batch = manager.batch()
        batch.track_entity("e1", source="doc")
        assert await batch.commit() is None


class TestUtility:
    async def test_check_reports_missing_references(self, manager):
        await manager.track_entity("e1", source="root", parent_entity_id="ghost")
        report = await manager.check()
        assert report["valid"] is False
        assert "e1 -> ghost" in report["missing_references"]

    async def test_statistics_and_clear(self, manager):
        await manager.track_entity("e1", source="doc-a")
        await manager.track_relationship("rel:a:knows:b", source="chunk-1")
        stats = await manager.get_statistics()
        assert stats["total_entries"] == 2
        assert stats["entity_types"] == {"entity": 1, "relationship": 1}
        assert stats["unique_sources"] == 2

        assert await manager.clear() == 2
        assert (await manager.get_statistics())["total_entries"] == 0

    async def test_tracking_degrades_gracefully_on_storage_failure(self, manager, monkeypatch):
        async def explode(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(storage, "append_chained", explode)
        assert await manager.track_entity("e1", source="doc") is None
        assert await manager.track_relationship("rel:a:b:c", source="doc") is None
        assert await manager.track_chunk("chunk-1", source_document="doc") is None
