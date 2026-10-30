"""Exercise archive inspection through the command and real CLI process."""

import argparse
import json
import os
import subprocess
import sys

import pytest

from cognee.cli.commands.inspect_archive_command import InspectArchiveCommand
from cognee.modules.migration.cogx import COGXArchiveWriter, COGXMemory


@pytest.fixture
def archive(tmp_path):
    directory = tmp_path / "archive"
    with COGXArchiveWriter(directory, source_system="cli-test") as writer:
        writer.write(COGXMemory(external_id="memory-1", content="A portable memory"))
    return directory


def _args(directory, *, as_json=False):
    parser = argparse.ArgumentParser()
    InspectArchiveCommand().configure_parser(parser)
    return parser.parse_args([str(directory), *(["--json"] if as_json else [])])


def test_text_report_shows_counts_and_checksum_coverage(archive, capsys):
    InspectArchiveCommand().execute(_args(archive))

    output = capsys.readouterr().out
    assert "Source system: cli-test" in output
    assert "memory: 1" in output
    assert "valid=1" in output
    assert "Inspection passed." in output
    assert "A portable memory" not in output


def test_json_report_is_one_structured_document(archive, capsys):
    InspectArchiveCommand().execute(_args(archive, as_json=True))

    report = json.loads(capsys.readouterr().out)
    assert report["valid"]
    assert report["record_counts"] == {"memory": 1}
    assert report["checksum_counts"] == {"valid": 1, "invalid": 0, "missing": 0}


def test_invalid_json_report_has_failing_exit_status_and_line_location(archive, capsys):
    (archive / "memories.jsonl").write_text("not json\n", encoding="utf-8")

    with pytest.raises(SystemExit) as failure:
        InspectArchiveCommand().execute(_args(archive, as_json=True))

    assert failure.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"]
    assert any(
        item["file"] == "memories.jsonl" and item["line"] == 1 for item in report["diagnostics"]
    )


def test_text_report_flags_account_data_without_printing_it(archive, capsys):
    (archive / "permissions.json").write_text(
        '{"email":"private@example.test","hashed_password":"secret-hash"}', encoding="utf-8"
    )

    InspectArchiveCommand().execute(_args(archive))

    output = capsys.readouterr().out
    assert "permissions.json is present" in output
    assert "private@example.test" not in output
    assert "secret-hash" not in output


@pytest.mark.parametrize("damaged", [False, True], ids=["valid", "damaged"])
def test_real_cli_reports_integrity_locally_without_creating_a_database(archive, tmp_path, damaged):
    if damaged:
        path = archive / "memories.jsonl"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["content"] = "Changed after export"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    system_root = tmp_path / "system"
    data_root = tmp_path / "data"
    environment = {
        **os.environ,
        "SYSTEM_ROOT_DIRECTORY": str(system_root),
        "DATA_ROOT_DIRECTORY": str(data_root),
        "TELEMETRY_DISABLED": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "COGNEE_CLI_MODE": "true",
        "LOG_LEVEL": "ERROR",
    }
    # The real dispatcher must keep this filesystem operation local even
    # when an API server is configured. Port 9 has no test service.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cognee.cli._cognee import main; sys.exit(main())",
            "--api-url",
            "http://127.0.0.1:9",
            "inspect-archive",
            str(archive),
            "--json",
        ],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
        cwd=tmp_path,
        timeout=45,
    )

    assert completed.returncode == (1 if damaged else 0), completed.stderr
    report = json.loads(completed.stdout)
    assert report["valid"] is not damaged
    assert report["checksum_counts"]["invalid"] == int(damaged)
    assert {p.name: p.read_bytes() for p in archive.iterdir()} == before
    # Normal logging can create directories, but no store may be initialized.
    for root in (system_root, data_root):
        assert not list(root.rglob("*databases*"))
        assert not list(root.rglob("*.db"))
        assert not list(root.rglob("*.sqlite*"))
        assert not list(root.rglob("*.lbug*"))
