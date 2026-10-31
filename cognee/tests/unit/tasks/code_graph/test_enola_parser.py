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
    assert receipt["enola_version"] == "0.4.12"
    assert receipt["format_version"] == 1
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


def _write_minimal_snapshot(tmp_path, insights):
    (tmp_path / "facts.jsonl").write_text(
        '{"kind": "symbol", "name": "app/db.Database", "file": "app/db.py"}\n'
    )
    (tmp_path / "insights.json").write_text(json.dumps(insights))


def test_parse_enola_snapshot_synthesizes_insight_facts(tmp_path):
    _write_minimal_snapshot(
        tmp_path,
        [
            {
                "title": "Call-graph hotspot: app/db.Database",
                "source": "hotspots",
                "confidence": 0.7,
                "description": "A pinch point.",
                "evidence": [
                    {"symbol": "app/db.Database", "detail": "fan-in 9"},
                    {"detail": "no target here"},
                ],
            }
        ],
    )

    facts, _receipt = parse_enola_snapshot(tmp_path)

    insight_facts = [fact for fact in facts if fact["kind"] == "insight"]
    assert len(insight_facts) == 1
    insight = insight_facts[0]
    assert insight["name"] == "Call-graph hotspot: app/db.Database"
    assert insight["props"]["source"] == "hotspots"
    assert insight["props"]["confidence"] == 0.7
    assert insight["relations"] == [{"kind": "evidences", "target": "app/db.Database"}]


def test_parse_enola_snapshot_tolerates_bad_insights(tmp_path):
    _write_minimal_snapshot(tmp_path, ["not-a-dict", {"no_title": True}])

    facts, _receipt = parse_enola_snapshot(tmp_path)

    assert [fact["kind"] for fact in facts] == ["symbol"]


def test_parse_enola_snapshot_ignores_corrupt_insights_file(tmp_path):
    (tmp_path / "facts.jsonl").write_text('{"kind": "module", "name": "app"}\n')
    (tmp_path / "insights.json").write_text("{ not json")

    facts, _receipt = parse_enola_snapshot(tmp_path)

    assert len(facts) == 1
