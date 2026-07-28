import json
from pathlib import Path

import pytest

from cognee.tasks.code_graph.enola import (
    EnolaSnapshotError,
    normalize_relation,
    parse_enola_snapshot,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_enola_snapshot_reads_all_valid_facts():
    facts, _receipt = parse_enola_snapshot(FIXTURES_DIR)

    # The fixture has 16 valid facts, one corrupt line, and one blank line.
    assert len(facts) == 16
    assert all(isinstance(fact, dict) for fact in facts)
    assert {fact["kind"] for fact in facts} == {
        "module",
        "symbol",
        "route",
        "storage",
        "dependency",
        "service",
    }


def test_parse_enola_snapshot_skips_corrupt_line():
    facts, _receipt = parse_enola_snapshot(FIXTURES_DIR)

    names = [fact["name"] for fact in facts]
    # Facts before and after the corrupt line both survive.
    assert "main" in names
    assert "helper" in names
    assert not any("not valid json" in json.dumps(fact) for fact in facts)


def test_parse_enola_snapshot_reads_receipt():
    _facts, receipt = parse_enola_snapshot(FIXTURES_DIR)

    assert receipt is not None
    assert receipt["enola_version"] == "0.3.1"
    assert receipt["snapshot_id"] == "sha256:abc123def456"


def test_parse_enola_snapshot_missing_receipt_is_not_fatal(tmp_path):
    (tmp_path / "facts.jsonl").write_text('{"kind": "module", "name": "app"}\n')

    facts, receipt = parse_enola_snapshot(tmp_path)

    assert len(facts) == 1
    assert receipt is None


def test_parse_enola_snapshot_missing_facts_raises(tmp_path):
    with pytest.raises(EnolaSnapshotError):
        parse_enola_snapshot(tmp_path)


@pytest.mark.parametrize(
    "relation, expected",
    [
        ({"type": "calls", "target": "helper"}, ("calls", "helper")),
        ({"kind": "imports", "to": "utils"}, ("imports", "utils")),
        ({"rel": "calls", "target_name": "helper"}, ("calls", "helper")),
        ({"relation": "implements", "name": "Storer"}, ("implements", "Storer")),
        ({"foo": "bar"}, None),
        ({"type": "calls"}, None),
        ({"target": "helper"}, None),
        ("not a dict", None),
    ],
)
def test_normalize_relation_tolerates_alternate_key_spellings(relation, expected):
    assert normalize_relation(relation) == expected
