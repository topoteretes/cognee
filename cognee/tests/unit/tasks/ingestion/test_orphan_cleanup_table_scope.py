"""Unit tests for table-scoped DLT orphan cleanup."""

from uuid import UUID

from cognee.tasks.ingestion.resolve_dlt_sources import _is_dlt_orphan_candidate


def _meta(table_name: str) -> dict:
    return {"source": "dlt", "table_name": table_name}


def test_orphan_candidate_when_table_in_run_and_id_missing():
    fresh_ids = {UUID("11111111-1111-1111-1111-111111111111")}
    stale_id = UUID("22222222-2222-2222-2222-222222222222")

    assert _is_dlt_orphan_candidate(
        _meta("slack_messages"), stale_id, fresh_ids, {"slack_messages"}
    )


def test_not_orphan_when_table_not_in_current_run():
    stale_id = UUID("22222222-2222-2222-2222-222222222222")

    assert not _is_dlt_orphan_candidate(
        _meta("employees"),
        stale_id,
        set(),
        {"slack_messages"},
    )


def test_not_orphan_when_id_still_fresh():
    data_id = UUID("11111111-1111-1111-1111-111111111111")

    assert not _is_dlt_orphan_candidate(
        _meta("slack_messages"),
        data_id,
        {data_id},
        {"slack_messages"},
    )


def test_not_orphan_for_non_dlt_source():
    stale_id = UUID("22222222-2222-2222-2222-222222222222")

    assert not _is_dlt_orphan_candidate(
        {"source": "file"},
        stale_id,
        set(),
        {"slack_messages"},
    )
