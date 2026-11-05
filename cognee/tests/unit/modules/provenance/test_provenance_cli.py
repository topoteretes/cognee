"""``cognee-cli provenance``: parses, prints, exits non-zero on a broken ledger."""

import argparse
import json
from uuid import uuid4

import pytest

from cognee.cli.commands.provenance_command import ProvenanceCommand
from cognee.cli.exceptions import CliCommandException
from cognee.modules.provenance.tombstones import ledger_edge_key, ledger_node_key

pytestmark = pytest.mark.asyncio


def _parse(*argv):
    parser = argparse.ArgumentParser()
    ProvenanceCommand().configure_parser(parser)
    return parser.parse_args(list(argv))


async def _seed(manager):
    ds = uuid4()
    batch = manager.batch()
    batch.track_entity(ledger_node_key(ds, "n1"), source="doc-a")
    batch.track_entity(ledger_node_key(ds, "n2"), source="doc-a")
    batch.track_relationship(
        ledger_edge_key(ds, "n1", "n2", "knows"),
        source="doc-a",
        used_entities=[ledger_node_key(ds, "n1"), ledger_node_key(ds, "n2")],
    )
    await batch.commit()
    await manager.invalidate(ledger_node_key(ds, "n2"), "auditor", reason="retracted")
    return ds


class TestParser:
    def test_subcommands_and_scope_flags(self):
        args = _parse("verify", "--dataset-id", str(uuid4()), "-f", "json")
        assert args.provenance_action == "verify"
        assert args.output_format == "json"
        assert _parse("lineage", "ds:n1").entity_id == "ds:n1"
        export = _parse("export", "-d", "docs", "--current-only", "-o", "out.jsonl")
        assert (export.dataset, export.current_only, export.output) == ("docs", True, "out.jsonl")

    def test_no_action_is_an_error(self):
        with pytest.raises(CliCommandException):
            ProvenanceCommand().execute(_parse())

    def test_registered_with_the_cli(self):
        from cognee.cli._cognee import _discover_commands

        assert ProvenanceCommand in _discover_commands()


class TestActions:
    async def test_verify_stats_check_scoped_to_dataset(self, manager, capsys):
        ds = await _seed(manager)
        command = ProvenanceCommand()

        await command._verify(_parse("verify", "--dataset-id", str(ds), "-f", "json"))
        verify = json.loads(capsys.readouterr().out)
        assert verify["valid"] is True and verify["total_entries"] == 4

        await command._stats(_parse("stats", "--dataset-id", str(ds), "-f", "json"))
        stats = json.loads(capsys.readouterr().out)
        assert (stats["live_entries"], stats["archived_entries"], stats["invalidated_count"]) == (
            3,
            1,
            1,
        )

        await command._check(_parse("check", "--dataset-id", str(uuid4()), "-f", "json"))
        assert json.loads(capsys.readouterr().out)["total_entries"] == 0

    async def test_verify_exits_2_on_tampering(self, manager, tamper):
        ds = await _seed(manager)
        await tamper(
            "UPDATE provenance_entries SET source_document = 'forged' WHERE entity_id = :id",
            {"id": ledger_node_key(ds, "n1")},
        )
        with pytest.raises(CliCommandException) as raised:
            await ProvenanceCommand()._verify(_parse("verify", "--dataset-id", str(ds)))
        assert raised.value.error_code == 2

    async def test_history_lineage_entry(self, manager, capsys):
        ds = await _seed(manager)
        command = ProvenanceCommand()

        await command._history(_parse("history", ledger_node_key(ds, "n2"), "-f", "json"))
        history = json.loads(capsys.readouterr().out)
        assert history[-1]["invalidated"] is True

        await command._lineage(_parse("lineage", ledger_edge_key(ds, "n1", "n2", "knows")))
        out = capsys.readouterr().out
        assert "Entries:    3" in out and "verified" in out

        await command._entry(_parse("entry", ledger_node_key(ds, "n1")))
        assert json.loads(capsys.readouterr().out)["entity_id"] == ledger_node_key(ds, "n1")

        with pytest.raises(CliCommandException):
            await command._entry(_parse("entry", ledger_node_key(ds, "missing")))

    async def test_export_writes_jsonl(self, manager, tmp_path):
        ds = await _seed(manager)
        out = tmp_path / "ledger.jsonl"
        await ProvenanceCommand()._export(_parse("export", "--dataset-id", str(ds), "-o", str(out)))
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 4
        assert [r["sequence_id"] for r in rows] == sorted(r["sequence_id"] for r in rows)

    async def test_anchor_and_verify_with_anchors(self, manager, monkeypatch, tmp_path, capsys):
        from types import SimpleNamespace

        from cognee.modules.provenance import anchors

        config = SimpleNamespace(
            provenance_anchor_key="k", provenance_anchor_path=str(tmp_path / "a.jsonl")
        )
        monkeypatch.setattr(anchors, "get_cognify_config", lambda: config)
        await _seed(manager)
        command = ProvenanceCommand()

        await command._anchor(_parse("anchor", "-f", "json"))
        anchor = json.loads(capsys.readouterr().out)["anchor"]
        assert anchor["sequence_id"] == 4

        await command._verify(_parse("verify", "--anchors", "-f", "json"))
        result = json.loads(capsys.readouterr().out)
        assert result["valid"] is True and result["anchors"]["valid"] is True

        config.provenance_anchor_key = None
        with pytest.raises(CliCommandException) as raised:
            await command._anchor(_parse("anchor"))
        assert raised.value.error_code == 1

    async def test_drift_needs_a_dataset_and_exits_2_on_drift(self, manager, monkeypatch, capsys):
        from types import SimpleNamespace

        from cognee.modules.provenance.manager import ProvenanceManager

        command = ProvenanceCommand()
        with pytest.raises(CliCommandException) as raised:
            await command._drift(_parse("drift"))
        assert raised.value.error_code == 1

        ds, owner = uuid4(), uuid4()

        async def _user(user_id=None):
            return SimpleNamespace(id=owner)

        async def _dataset(user, dataset_id, permission_type="read"):
            return SimpleNamespace(id=dataset_id, owner_id=owner)

        monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", _user, raising=False)
        monkeypatch.setattr(
            "cognee.modules.data.methods.get_authorized_dataset", _dataset, raising=False
        )

        async def _drift(self, dataset_id, owner_id=None):
            return {
                "valid": False,
                "dataset_id": str(dataset_id),
                "checked": 2,
                "unsnapshotted": 0,
                "drifted": [{"entity_id": f"{ds}:n1", "delta": {"description": ["a", "b"]}}],
                "missing_in_graph": [f"{ds}:n2"],
            }

        monkeypatch.setattr(ProvenanceManager, "check_drift", _drift)
        with pytest.raises(CliCommandException) as raised:
            await command._drift(_parse("drift", "--dataset-id", str(ds)))
        assert raised.value.error_code == 2
        out = capsys.readouterr().out
        assert "changed" in out and "missing" in out and "description" in out

    async def test_conflicting_scope_flags(self, manager):
        with pytest.raises(CliCommandException):
            await ProvenanceCommand()._stats(
                _parse("stats", "-d", "docs", "--dataset-id", str(uuid4()))
            )
        with pytest.raises(CliCommandException):
            await ProvenanceCommand()._stats(_parse("stats", "--dataset-id", "not-a-uuid"))
