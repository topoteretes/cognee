"""``cognee-cli provenance``: parses, prints, exits non-zero on a broken ledger."""

import argparse
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.cli.commands.provenance_command import ProvenanceCommand
from cognee.cli.exceptions import CliCommandException
from cognee.modules.provenance import access
from cognee.modules.provenance.tombstones import ledger_edge_key, ledger_node_key

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def cli_user(monkeypatch):
    """The user the CLI acts as: a superuser, as the default user is.

    A test narrows it by flipping ``is_superuser`` and listing ``readable`` datasets.
    """
    user = SimpleNamespace(id=uuid4(), is_superuser=True, readable=set())

    async def _resolve(user_id=None, strict=False):
        assert strict, "an unknown --user-id must not fall back to the default superuser"
        return user

    async def _authorized(acting_user, dataset_id, permission_type="read"):
        allowed = acting_user.is_superuser or dataset_id in acting_user.readable
        return SimpleNamespace(id=dataset_id) if allowed else None

    monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", _resolve)
    monkeypatch.setattr(access, "get_authorized_dataset", _authorized)
    return user


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

    async def test_a_user_reads_only_the_datasets_they_may_read(self, manager, cli_user, capsys):
        ds = await _seed(manager)
        cli_user.is_superuser = False
        command = ProvenanceCommand()
        key = ledger_node_key(ds, "n1")

        denied = (
            (command._stats, _parse("stats", "--dataset-id", str(ds))),
            (command._verify, _parse("verify", "--dataset-id", str(ds))),
            (command._check, _parse("check", "--dataset-id", str(ds))),
            (command._export, _parse("export", "--dataset-id", str(ds))),
            (command._history, _parse("history", key)),
            (command._entry, _parse("entry", key)),
            (command._lineage, _parse("lineage", key)),
        )
        for handler, args in denied:
            with pytest.raises(CliCommandException) as raised:
                await handler(args)
            assert raised.value.error_code == 1
        assert "No read access" in capsys.readouterr().err

        cli_user.readable.add(ds)
        for handler, args in denied:
            await handler(args)

    async def test_the_whole_ledger_and_anchoring_are_superuser_only(self, manager, cli_user):
        await _seed(manager)
        cli_user.is_superuser = False
        command = ProvenanceCommand()
        for handler, args in (
            (command._stats, _parse("stats")),
            (command._verify, _parse("verify")),
            (command._verify, _parse("verify", "--anchors")),
            (command._check, _parse("check")),
            (command._export, _parse("export")),
            (command._anchor, _parse("anchor")),
        ):
            with pytest.raises(CliCommandException) as raised:
                await handler(args)
            assert raised.value.error_code == 1

    async def test_conflicting_scope_flags(self, manager):
        with pytest.raises(CliCommandException):
            await ProvenanceCommand()._stats(
                _parse("stats", "-d", "docs", "--dataset-id", str(uuid4()))
            )
        with pytest.raises(CliCommandException):
            await ProvenanceCommand()._stats(_parse("stats", "--dataset-id", "not-a-uuid"))
