import argparse
from unittest.mock import AsyncMock, patch

import pytest

from cognee.cli.commands.remember_command import RememberCommand
from cognee.tasks.presort.models import PresortReport


@pytest.fixture
def command_parser():
    command = RememberCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    return command, parser


@pytest.mark.parametrize("argv", [["--dry-run", "notes.txt"], ["notes.txt", "--dry-run"]])
def test_dry_run_does_not_consume_data(command_parser, argv):
    command, parser = command_parser
    args = parser.parse_args(argv)
    assert args.data == ["notes.txt"]
    assert args.dry_run is True
    assert args.presort is False
    with patch("cognee.remember", new=AsyncMock(return_value="estimate")) as remember:
        command.execute(args)
    assert remember.await_args.kwargs["data"] == "notes.txt"
    assert remember.await_args.kwargs["dry_run"] is True


@pytest.mark.parametrize("argv", [["--presort", "folder"], ["folder", "--presort"]])
def test_presort_routes_to_explicit_scan(command_parser, argv, tmp_path):
    command, parser = command_parser
    args = parser.parse_args(argv)
    report = PresortReport(scan_id="s", root_path=str(tmp_path))
    with patch("cognee.remember", new=AsyncMock(return_value=report)) as remember:
        command.execute(args)
    assert remember.await_args.kwargs["data"] == "folder"
    assert remember.await_args.kwargs["dry_run"] == "presort"
    assert remember.await_args.kwargs["auto_apply"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--presort", "--dry-run", "folder"],
        ["--dry-run", "--from-report", "report.json"],
        ["--presort", "--from-report", "report.json"],
    ],
)
def test_conflicting_modes_rejected(command_parser, argv):
    _, parser = command_parser
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_allow_root_preserves_unrestricted_default(command_parser, monkeypatch, tmp_path):
    import os

    command, _ = command_parser
    monkeypatch.delenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", raising=False)
    command._extend_allowed_roots([tmp_path])
    assert "COGNEE_ALLOWED_LOCAL_FILE_ROOTS" not in os.environ


def test_apply_report_can_extend_existing_allowlist(command_parser, monkeypatch, tmp_path):
    import os

    command, parser = command_parser
    allowed = tmp_path / "allowed"
    source = tmp_path / "source"
    report = PresortReport(scan_id="s", root_path=str(source))
    saved = report.save(tmp_path / "report.presort.json")
    monkeypatch.setenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", str(allowed))
    args = parser.parse_args(["--from-report", saved, "--allow-root"])
    with patch("cognee.remember", new=AsyncMock(return_value={})) as remember:
        command.execute(args)
    assert remember.await_args.kwargs["data"] == report
    assert os.environ["COGNEE_ALLOWED_LOCAL_FILE_ROOTS"].split(os.pathsep) == [
        str(allowed),
        saved,
        str(source),
    ]
