import json

import pytest

from cognee.context_global_variables import current_dataset_id
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


@pytest.mark.parametrize(
    "date,expected",
    [(None, "[accepted by user]"), ("2024-01-01", "[accepted by user; as of 2024-01-01]")],
)
def test_a_fact_the_user_selected_says_so_over_any_review(date, expected):
    marks = [
        {"conflict_id": "ceo", "status": "current", "resolution_source": "user", "dataset_id": "a"},
        {"conflict_id": "employer", "status": "conflicting"},
    ]
    properties = {"effective_date": date, "conflict_marks": marks}
    assert status_label(properties) == expected
    # Another dataset's decision is not this search's: the review's status shows.
    token = current_dataset_id.set("b")
    try:
        assert fact_status(properties) == "conflicting"
    finally:
        current_dataset_id.reset(token)
