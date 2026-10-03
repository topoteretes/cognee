"""External HMAC anchors: the defense against rewrite-and-rechain tampering."""

import json
from uuid import uuid4

import pytest

from cognee.modules.provenance import anchors
from cognee.modules.provenance.integrity import compute_checksum
from cognee.modules.provenance.tombstones import ledger_node_key

pytestmark = pytest.mark.asyncio


@pytest.fixture
def anchor_config(monkeypatch, tmp_path):
    """Point anchoring at a temp file with a known key."""
    from types import SimpleNamespace

    path = tmp_path / "anchors.jsonl"
    config = SimpleNamespace(provenance_anchor_key="test-secret", provenance_anchor_path=str(path))
    monkeypatch.setattr(anchors, "get_cognify_config", lambda: config)
    return config, path


async def _seed(manager, n=3):
    ds = uuid4()
    batch = manager.batch()
    for index in range(n):
        batch.track_entity(ledger_node_key(ds, f"n{index}"), source="doc")
    await batch.commit()
    return ds


class TestConfiguration:
    async def test_unconfigured_key_raises(self, manager, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(
            anchors,
            "get_cognify_config",
            lambda: SimpleNamespace(provenance_anchor_key=None, provenance_anchor_path=None),
        )
        assert anchors.anchoring_configured() is False
        with pytest.raises(anchors.AnchoringNotConfiguredError):
            await manager.anchor()

    def test_default_path_is_under_system_root(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(
            anchors,
            "get_cognify_config",
            lambda: SimpleNamespace(provenance_anchor_key="k", provenance_anchor_path=None),
        )
        monkeypatch.setattr(
            anchors, "get_base_config", lambda: SimpleNamespace(system_root_directory="/sys")
        )
        assert anchors.anchor_path() == "/sys/provenance_anchors.jsonl"


class TestAnchorAndVerify:
    async def test_empty_ledger_has_nothing_to_anchor(self, manager, anchor_config):
        assert await manager.anchor() is None
        result = await manager.verify_anchors()
        assert result == {"valid": True, "anchored": False, "anchors_checked": 0, "failures": []}

    async def test_anchor_appends_signed_head_and_verifies(self, manager, anchor_config):
        _, path = anchor_config
        await _seed(manager, 3)
        first = await manager.anchor()
        assert first["sequence_id"] == 3 and first["total_entries"] == 3
        assert anchors.signature_valid(first)

        await _seed(manager, 2)
        second = await manager.anchor()
        assert second["sequence_id"] == 5

        lines = path.read_text().splitlines()
        assert [json.loads(line)["sequence_id"] for line in lines] == [3, 5]

        result = await manager.verify_anchors()
        assert result["valid"] is True
        assert result["anchors_checked"] == 2
        assert result["latest_anchor"]["sequence_id"] == 5

    async def test_rewrite_and_rechain_is_caught_only_by_the_anchor(self, manager, anchor_config):
        """The attack a bare hash chain cannot see: alter a row, then recompute
        its checksum and every later previous_checksum so verify_chain passes."""
        from sqlalchemy import select, update

        from cognee.modules.provenance import storage
        from cognee.modules.provenance.models import ProvenanceEntry, ProvenanceEntryRow

        ds = await _seed(manager, 3)
        await manager.anchor()

        async with storage.get_async_session() as session, session.begin():
            rows = (
                (
                    await session.execute(
                        select(ProvenanceEntryRow).order_by(ProvenanceEntryRow.sequence_id)
                    )
                )
                .scalars()
                .all()
            )
            previous = None
            for row in rows:
                if row.entity_id == ledger_node_key(ds, "n0"):
                    row.source_document = "forged"
                row.previous_checksum = previous
                entry = ProvenanceEntry.from_row(row)
                entry.previous_checksum = previous
                row.checksum = compute_checksum(entry)
                previous = row.checksum
                await session.execute(
                    update(ProvenanceEntryRow)
                    .where(ProvenanceEntryRow.entity_id == row.entity_id)
                    .values(
                        source_document=row.source_document,
                        previous_checksum=row.previous_checksum,
                        checksum=row.checksum,
                    )
                )

        # Internally consistent again...
        assert (await manager.verify_chain())["valid"] is True
        # ...but the anchored head no longer matches.
        result = await manager.verify_anchors()
        assert result["valid"] is False
        assert result["failures"][0]["reason"] == "checksum_mismatch"

    async def test_forged_anchor_and_deleted_rows_are_caught(self, manager, anchor_config):
        _, path = anchor_config
        await _seed(manager, 2)
        anchor = await manager.anchor()

        forged = dict(anchor, checksum="0" * 64)  # signature no longer matches payload
        missing = anchors.build_anchor(99, "a" * 64, 99)  # valid signature, no such row
        result = await anchors.verify_anchors([anchor, forged, missing])
        assert result["valid"] is False
        assert [f["reason"] for f in result["failures"]] == ["bad_signature", "missing_row"]

        path.write_text("not json\n")
        result = await manager.verify_anchors()
        assert result["failures"] == [{"line": 1, "reason": "malformed"}]

    async def test_key_rotation_invalidates_old_signatures(self, manager, anchor_config):
        config, _ = anchor_config
        await _seed(manager, 1)
        await manager.anchor()
        config.provenance_anchor_key = "rotated"
        result = await manager.verify_anchors()
        assert [f["reason"] for f in result["failures"]] == ["bad_signature"]
