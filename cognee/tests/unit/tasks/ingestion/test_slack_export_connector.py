"""Unit tests for the Slack workspace export DLT connector."""

import os

import pytest

from cognee.tasks.ingestion.connectors.slack_export import (
    iter_slack_export_messages,
    slack_export_source,
)

FIXTURES = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "examples",
        "demos",
        "test_data",
        "slack_export",
    )
)


def _fixture(name: str) -> str:
    return os.path.normpath(os.path.join(FIXTURES, name))


def test_iter_messages_v1_yields_expected_rows():
    rows = list(iter_slack_export_messages(_fixture("v1")))

    assert len(rows) == 4
    ids = {row["id"] for row in rows}
    assert "C001:1717200000.000100" in ids
    assert "C002:1717200100.000100" in ids


def test_message_ids_use_channel_id_and_ts():
    rows = list(iter_slack_export_messages(_fixture("v1")))
    for row in rows:
        channel_id, ts = row["id"].split(":", 1)
        assert channel_id == row["channel_id"]
        assert ts == row["ts"]


def test_thread_reply_text_includes_thread_marker():
    rows = list(iter_slack_export_messages(_fixture("v1")))
    reply = next(row for row in rows if row["ts"] == "1717200001.000200")

    assert reply["thread_ts"] == "1717200000.000100"
    assert "[thread reply" in reply["text"]
    assert "Bob" in reply["text"]


def test_v2_deleted_message_would_be_orphan_on_resync():
    from cognee.tasks.ingestion.resolve_dlt_sources import _is_dlt_orphan_candidate

    v1_ids = {row["id"] for row in iter_slack_export_messages(_fixture("v1"))}
    v2_ids = {row["id"] for row in iter_slack_export_messages(_fixture("v2"))}
    deleted_ids = v1_ids - v2_ids

    assert deleted_ids == {"C001:1717200002.000300"}
    fresh_table_names = {"slack_messages"}
    for deleted_id in deleted_ids:
        assert _is_dlt_orphan_candidate(
            {"source": "dlt", "table_name": "slack_messages"},
            "00000000-0000-0000-0000-000000000001",
            set(),
            fresh_table_names,
        )


def test_large_export_has_more_than_default_row_cap():
    rows = list(iter_slack_export_messages(_fixture("large")))
    assert len(rows) == 65


def test_missing_channels_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="channels.json"):
        list(iter_slack_export_messages(tmp_path))


pytest.importorskip("dlt")


def test_slack_export_source_is_dlt_resource():
    resource = slack_export_source(_fixture("v1"))
    rows = list(resource)

    assert len(rows) == 4
    assert all("id" in row for row in rows)


def test_large_fixture_exceeds_default_dlt_row_cap():
    from cognee.tasks.ingestion.config import get_ingestion_config

    rows = list(iter_slack_export_messages(_fixture("large")))
    default_cap = get_ingestion_config().dlt_max_rows_per_table

    assert default_cap == 50
    assert len(rows) > default_cap
