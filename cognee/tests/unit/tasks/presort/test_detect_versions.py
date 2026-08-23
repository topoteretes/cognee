import pytest

from cognee.tasks.presort.detect_versions import detect_versions, normalize_stem
from cognee.tasks.presort.models import FileRecord


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("report (1)", "report"),
        ("report (12)", "report"),
        ("report copy", "report"),
        ("report copy 2", "report"),
        ("Report-FINAL", "report"),
        ("report_v2", "report"),
        ("report-3", "report"),
        ("report_2024-01-15", "report"),
        ("report_20240115", "report"),
        ("report_v2 (1)", "report"),  # stacked suffixes
        ("plain", "plain"),
    ],
)
def test_normalize_stem(stem, expected):
    assert normalize_stem(stem) == expected


def _record(path: str, content_hash: str) -> FileRecord:
    name = path.rsplit("/", 1)[-1]
    extension = name.rsplit(".", 1)[-1] if "." in name else ""
    return FileRecord(path=path, name=name, extension=extension, content_hash=content_hash)


def test_versions_require_differing_content():
    same = [_record("/d/report.pdf", "h1"), _record("/d/report (1).pdf", "h1")]
    assert detect_versions(same) == []  # identical content = duplicates, not versions

    different = [_record("/d/report.pdf", "h1"), _record("/d/report_v2.pdf", "h2")]
    candidates = detect_versions(different)
    assert len(candidates) == 1
    assert candidates[0].normalized_stem == "report"


def test_versions_grouped_per_directory_and_extension():
    records = [
        _record("/d1/report.pdf", "h1"),
        _record("/d2/report_v2.pdf", "h2"),  # other directory: no pair
        _record("/d1/report.txt", "h3"),  # other extension: no pair
    ]
    assert detect_versions(records) == []
