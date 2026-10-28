import json

import pytest

from cognee.modules.graph.utils.fact_conflicts import fact_status, status_label


@pytest.mark.parametrize("json_alias", [False, True])
@pytest.mark.parametrize(
    "statuses,date,expected",
    [
        ([], "2026-01-10", ""),
        (["current"], None, "[current]"),
        (["superseded"], None, "[superseded]"),
        (["conflicting"], None, "[conflicting]"),
        (["current"], "2026-01-10T08:00:00+00:00", "[as of 2026-01-10]"),
        (["superseded", "current"], "2020-05-01", "[superseded; as of 2020-05-01]"),
        (["current", "superseded", "conflicting"], "2020-05-01", "[conflicting; as of 2020-05-01]"),
    ],
)
def test_labels_and_precedence(statuses, date, expected, json_alias):
    marks = [{"conflict_id": str(index), "status": status} for index, status in enumerate(statuses)]
    properties = {"effective_date": date}
    properties["conflict_marks_json" if json_alias else "conflict_marks"] = (
        json.dumps(marks) if json_alias else marks
    )
    assert status_label(properties) == expected
    if not statuses:
        assert fact_status(properties) is None
