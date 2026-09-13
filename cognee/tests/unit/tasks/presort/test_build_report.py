import pytest

from cognee.modules.graph_models import GraphSchemaSpec
from cognee.tasks.presort.build_report import build_report
from cognee.tasks.presort.default_spec import DEFAULT_PRESORT_SPEC
from cognee.tasks.presort.models import (
    DuplicateCluster,
    FileRecord,
    PiiFinding,
    PresortReport,
    ProposedGroup,
)


def _inputs(tmp_path):
    files = [
        FileRecord(path=str(tmp_path / "a.txt"), name="a.txt", size_bytes=5, cognee_status="new"),
        FileRecord(
            path=str(tmp_path / "b.txt"), name="b.txt", size_bytes=7, cognee_status="cognified"
        ),
    ]
    duplicates = [DuplicateCluster(content_hash="h", paths=["/a", "/b"], size_bytes=5)]
    pii = [PiiFinding(path=str(tmp_path / "a.txt"), category="email_address", severity="low")]
    groups = [ProposedGroup(name="docs", dataset_name="docs", file_paths=[files[0].path])]
    return files, duplicates, pii, groups


def test_default_spec_enables_all_sections(tmp_path):
    files, duplicates, pii, groups = _inputs(tmp_path)
    spec = GraphSchemaSpec.model_validate(DEFAULT_PRESORT_SPEC)

    report = build_report(
        tmp_path, files, [], duplicates, [], pii, groups, spec=spec, used_llm=False
    )

    assert report.duplicates == duplicates
    assert report.pii == pii
    assert report.groups == groups
    assert report.spec_used["entities"][0]["name"] == "FileRecord"
    assert report.scan_id  # stable id present

    summary = report.summary()
    assert summary["files"] == 2
    assert summary["cognee_status"]["new"] == 1
    assert summary["cognee_status"]["cognified"] == 1
    assert summary["bytes_needing_processing"] == 5
    assert summary["wasted_bytes"] == 5


def test_spec_without_pii_relation_drops_pii_section(tmp_path):
    files, duplicates, pii, groups = _inputs(tmp_path)
    spec_dict = {
        "root": "FileRecord",
        "entities": [
            {
                "name": "FileRecord",
                "fields": [
                    {
                        "kind": "relation",
                        "name": "duplicate_of",
                        "relation": {"target_entity_name": "FileRecord", "cardinality": "many"},
                    },
                    {
                        "kind": "relation",
                        "name": "belongs_to_group",
                        "relation": {"target_entity_name": "FileGroup", "cardinality": "one"},
                    },
                ],
            },
            {"name": "FileGroup"},
        ],
    }
    spec = GraphSchemaSpec.model_validate(spec_dict)

    report = build_report(
        tmp_path, files, [], duplicates, [], pii, groups, spec=spec, used_llm=False
    )

    assert report.pii == []
    assert report.versions == []
    assert report.duplicates == duplicates
    assert report.groups == groups
    assert any("'pii' disabled" in warning for warning in report.warnings)


def test_report_json_round_trip(tmp_path):
    files, duplicates, pii, groups = _inputs(tmp_path)
    spec = GraphSchemaSpec.model_validate(DEFAULT_PRESORT_SPEC)
    report = build_report(
        tmp_path, files, [], duplicates, [], pii, groups, spec=spec, used_llm=True
    )

    # dict round trip (with the marker key remember() detects)
    as_dict = report.to_dict()
    assert as_dict["presort_report"] is True
    assert PresortReport.from_json(as_dict) == report

    # file round trip
    saved_path = report.save(tmp_path / "reports" / "report.json")
    loaded = PresortReport.from_json(saved_path)
    assert loaded.scan_id == report.scan_id
    assert loaded.files == report.files
    assert loaded.report_path == saved_path


def test_scan_id_stable(tmp_path):
    files, _, _, _ = _inputs(tmp_path)
    spec = GraphSchemaSpec.model_validate(DEFAULT_PRESORT_SPEC)
    first = build_report(tmp_path, files, [], [], [], [], [], spec=spec)
    second = build_report(tmp_path, files, [], [], [], [], [], spec=spec)
    assert first.scan_id == second.scan_id


def test_report_json_text_does_not_probe_filesystem(tmp_path, monkeypatch):
    from cognee.modules.presort import local_paths

    def reject_path(*args, **kwargs):
        pytest.fail("JSON content must not be treated as a local path")

    monkeypatch.setattr(local_paths, "resolve_presort_path", reject_path)
    report = PresortReport(scan_id="s", root_path=str(tmp_path))
    assert PresortReport.from_json(" \n" + report.to_json()) == report


@pytest.mark.parametrize("path_kind", ["outside", "traversal", "symlink"])
def test_report_loading_rejects_paths_outside_allowed_roots(tmp_path, monkeypatch, path_kind):
    from cognee.modules.presort import local_paths

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = tmp_path / "outside.presort.json"
    PresortReport(scan_id="s", root_path=str(tmp_path)).save(source)
    if path_kind == "traversal":
        source = allowed / ".." / source.name
    elif path_kind == "symlink":
        link = allowed / "link.presort.json"
        try:
            link.symlink_to(source)
        except OSError:
            pytest.skip("Symlink creation is unavailable")
        source = link
    monkeypatch.setattr(local_paths, "get_presort_roots", lambda: (allowed,))
    with pytest.raises(ValueError, match="outside allowed roots"):
        PresortReport.from_json(source)
