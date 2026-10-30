"""Inspect real COGX writer output without importing or changing its contents."""

import json

import pytest

from cognee.migration import inspect_cogx_archive
from cognee.modules.migration.cogx import (
    COGX_VERSION,
    COGXArchiveWriter,
    COGXDocument,
    COGXEntity,
    COGXEpisode,
    COGXFact,
    COGXMemory,
    COGXMemoryBlock,
    COGXTurn,
)


@pytest.fixture
def archive_dir(tmp_path):
    directory = tmp_path / "archive"
    with COGXArchiveWriter(directory, source_system="inspection-test") as writer:
        writer.write(COGXDocument(external_id="doc-1", content="Source text"))
        writer.write(
            COGXEpisode(
                external_id="episode-1",
                turns=[COGXTurn(role="user", content="A conversation turn")],
            )
        )
        writer.write(COGXEntity(external_id="entity-1", name="Alice"))
        writer.write(
            COGXFact(
                external_id="fact-1",
                subject_ref="entity-1",
                predicate="lives_in",
                object_ref="Berlin",
            )
        )
        writer.write(COGXMemory(external_id="memory-1", content="A useful memory"))
        writer.write(COGXMemoryBlock(external_id="block-1", label="persona", value="Helpful"))
        # A raw node's 'kind' property is not a COGX discriminator.
        writer.write_raw_node({"id": "raw-1", "type": "NodeSet", "kind": "application-data"})
    return directory


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _diagnostics(report, code):
    return [item for item in report.diagnostics if item.code == code]


@pytest.mark.parametrize(
    "malformed_fields", [{"external_id": "\ud800"}, {"metadata": {"value": "\ud800"}}]
)
def test_unserializable_checksum_reports_line_and_continues(tmp_path, malformed_fields):
    invalid = {
        "kind": "document",
        "external_id": "invalid-record",
        "content": "safe text",
        "checksum": "sha256:bad",
        **malformed_fields,
    }
    valid = {"kind": "document", "external_id": "next-record", "content": "safe text"}
    (tmp_path / "documents.jsonl").write_text(
        json.dumps(invalid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8"
    )

    report = inspect_cogx_archive(tmp_path)

    assert not report.valid
    assert report.record_counts == {"document": 2}
    assert report.checksum_counts == {"valid": 0, "invalid": 1, "missing": 1}
    diagnostic = _diagnostics(report, "checksum_error")[0]
    assert diagnostic.file == "documents.jsonl"
    assert diagnostic.line == 1
    assert json.loads(report.model_dump_json())["valid"] is False


def test_inspect_writer_archive_counts_every_record_kind_and_verifies_checksums(archive_dir):
    report = inspect_cogx_archive(archive_dir)

    assert report.valid
    assert report.cogx_version == COGX_VERSION
    assert report.source_system == "inspection-test"
    assert report.record_counts == {
        "document": 1,
        "episode": 1,
        "entity": 1,
        "fact": 1,
        "memory": 1,
        "memory_block": 1,
        "raw_node": 1,
    }
    assert report.declared_counts == report.record_counts
    assert report.total_records == 7
    assert report.checksum_counts == {"valid": 6, "invalid": 0, "missing": 1}
    assert not report.contains_permissions


def test_inspection_does_not_change_archive_bytes_or_modification_times(archive_dir):
    def snapshot():
        return {
            path.relative_to(archive_dir): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in archive_dir.rglob("*")
            if path.is_file()
        }

    before = snapshot()

    inspect_cogx_archive(str(archive_dir))

    assert snapshot() == before


def test_inspection_detects_tampered_content_with_a_valid_record_schema(archive_dir):
    document_path = archive_dir / "documents.jsonl"
    document = _read_json(document_path)
    document["content"] = "Changed after the archive was written"
    _write_json(document_path, document)

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert report.record_counts["document"] == 1
    assert report.checksum_counts == {"valid": 5, "invalid": 1, "missing": 1}
    mismatch = _diagnostics(report, "checksum_mismatch")
    assert len(mismatch) == 1
    assert mismatch[0].file == "documents.jsonl"
    assert mismatch[0].line == 1
    assert mismatch[0].severity == "error"


def test_missing_manifest_is_a_warning_for_an_otherwise_valid_archive(archive_dir):
    (archive_dir / "manifest.json").unlink()

    report = inspect_cogx_archive(archive_dir)

    assert report.valid
    assert report.total_records == 7
    assert report.declared_counts == {}
    assert _diagnostics(report, "missing_manifest")[0].severity == "warning"


@pytest.mark.parametrize("content", ["{broken-json", '{"counts": {"document": "many"}}'])
def test_invalid_manifest_is_reported_while_records_are_still_inspected(archive_dir, content):
    (archive_dir / "manifest.json").write_text(content, encoding="utf-8")

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert report.total_records == 7
    problem = _diagnostics(report, "invalid_manifest")
    assert len(problem) == 1
    assert problem[0].file == "manifest.json"
    assert problem[0].severity == "error"


@pytest.mark.parametrize("version", [f"{int(COGX_VERSION.split('.')[0]) + 1}.0", "not-a-version"])
def test_unsupported_or_unrecognized_manifest_version_is_reported(archive_dir, version):
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["cogx_version"] = version
    _write_json(manifest_path, manifest)

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert _diagnostics(report, "unsupported_version")[0].severity == "error"


def test_declared_count_mismatch_is_reported(archive_dir):
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["counts"]["document"] = 3
    _write_json(manifest_path, manifest)

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert report.record_counts["document"] == 1
    assert report.declared_counts["document"] == 3
    assert _diagnostics(report, "count_mismatch")[0].severity == "error"


def test_manifest_counts_can_be_entirely_omitted_for_legacy_archives(archive_dir):
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    del manifest["counts"]
    _write_json(manifest_path, manifest)

    report = inspect_cogx_archive(archive_dir)

    assert report.valid
    assert report.total_records == 7
    assert report.declared_counts == {}
    assert not _diagnostics(report, "count_mismatch")


def test_explicitly_empty_manifest_counts_cannot_hide_observed_records(archive_dir):
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["counts"] = {}
    _write_json(manifest_path, manifest)

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert report.total_records == 7
    assert report.declared_counts == {}
    assert _diagnostics(report, "count_mismatch")


def test_undeclared_record_kind_is_a_mismatch_when_other_counts_are_declared(archive_dir):
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    del manifest["counts"]["raw_node"]
    _write_json(manifest_path, manifest)

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert report.record_counts["raw_node"] == 1
    assert report.declared_counts["document"] == 1
    assert "raw_node" not in report.declared_counts
    problems = _diagnostics(report, "count_mismatch")
    assert len(problems) == 1
    assert problems[0].file == "manifest.json"
    assert problems[0].severity == "error"


def test_missing_declared_record_file_is_a_count_mismatch(archive_dir):
    (archive_dir / "memories.jsonl").unlink()

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert report.record_counts.get("memory", 0) == 0
    assert report.total_records == 6
    assert _diagnostics(report, "count_mismatch")


def test_negative_manifest_count_cannot_make_an_archive_valid(archive_dir):
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["counts"]["document"] = -1
    _write_json(manifest_path, manifest)

    report = inspect_cogx_archive(archive_dir)

    assert not report.valid
    assert any(item.severity == "error" for item in report.diagnostics)


def test_validation_continues_after_invalid_json_and_invalid_record(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    good = COGXDocument(external_id="good", content="A valid later record")
    lines = [
        "{broken-json",
        json.dumps({"kind": "document", "external_id": "missing-content"}),
        "",
        good.model_dump_json(exclude_none=True),
    ]
    (directory / "documents.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = inspect_cogx_archive(directory)

    assert not report.valid
    assert report.record_counts == {"document": 1}
    assert report.total_records == 1
    assert report.checksum_counts == {"valid": 0, "invalid": 0, "missing": 1}
    assert _diagnostics(report, "invalid_json")[0].line == 1
    assert _diagnostics(report, "invalid_record")[0].line == 2
    assert all(item.file == "documents.jsonl" for item in report.diagnostics if item.line)


def test_schema_diagnostics_do_not_echo_private_record_values(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    (directory / "documents.jsonl").write_text(
        json.dumps(
            {
                "kind": "document",
                "external_id": "document-1",
                "content": ["PRIVATE-RECORD-TEXT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_cogx_archive(directory)

    assert not report.valid
    assert _diagnostics(report, "invalid_record")
    assert "PRIVATE-RECORD-TEXT" not in report.model_dump_json()


def test_record_kind_must_match_the_file_where_it_is_stored(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    entity = COGXEntity(external_id="entity-1", name="Alice")
    (directory / "documents.jsonl").write_text(entity.model_dump_json() + "\n", encoding="utf-8")

    report = inspect_cogx_archive(directory)

    assert not report.valid
    assert report.total_records == 0
    problem = _diagnostics(report, "wrong_record_kind")[0]
    assert problem.file == "documents.jsonl"
    assert problem.line == 1


def test_legacy_record_without_checksum_is_counted_as_unverified_not_invalid(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    (directory / "documents.jsonl").write_text(
        '{"kind":"document","external_id":"legacy","content":"Original text"}\n',
        encoding="utf-8",
    )

    report = inspect_cogx_archive(directory)

    assert report.valid
    assert report.total_records == 1
    assert report.checksum_counts == {"valid": 0, "invalid": 0, "missing": 1}


@pytest.mark.parametrize(
    "permissions", ['{"password_hash":"PRIVATE-HASH"}', "PRIVATE-HASH invalid JSON"]
)
def test_permissions_presence_is_reported_without_parsing_or_exposing_secrets(
    archive_dir, permissions
):
    (archive_dir / "permissions.json").write_text(permissions, encoding="utf-8")

    report = inspect_cogx_archive(archive_dir)

    assert report.valid
    assert report.contains_permissions
    assert "PRIVATE-HASH" not in report.model_dump_json()
    assert report.total_records == 7


@pytest.mark.parametrize("filename", ["documents.jsonl", "manifest.json", "permissions.json"])
def test_known_files_cannot_follow_symlinks_outside_the_archive(tmp_path, filename):
    directory = tmp_path / "archive"
    directory.mkdir()
    outside = tmp_path / "private.json"
    outside.write_text("PRIVATE-OUTSIDE-CONTENT", encoding="utf-8")
    try:
        (directory / filename).symlink_to(outside)
    except OSError:
        pytest.skip("File symlinks are not supported on this platform")

    report = inspect_cogx_archive(directory)

    assert not report.valid
    assert _diagnostics(report, "unsafe_file")[0].file == filename
    assert "PRIVATE-OUTSIDE-CONTENT" not in report.model_dump_json()
    assert outside.read_text(encoding="utf-8") == "PRIVATE-OUTSIDE-CONTENT"


def test_missing_directory_is_an_invalid_report_without_creating_it(tmp_path):
    missing = tmp_path / "does-not-exist"

    report = inspect_cogx_archive(missing)

    assert not report.valid
    assert _diagnostics(report, "directory_not_found")
    assert report.total_records == 0
    assert not missing.exists()


def test_unrelated_directory_is_not_an_archive(tmp_path):
    (tmp_path / "notes.txt").write_text("Ordinary folder", encoding="utf-8")

    report = inspect_cogx_archive(tmp_path)

    assert not report.valid
    assert _diagnostics(report, "not_archive")


def test_empty_archive_written_by_cogx_writer_is_valid(tmp_path):
    directory = tmp_path / "empty-archive"
    with COGXArchiveWriter(directory):
        pass

    report = inspect_cogx_archive(directory)

    assert report.valid
    assert report.total_records == 0
    assert report.record_counts == report.declared_counts == {}
